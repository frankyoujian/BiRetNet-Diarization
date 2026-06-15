#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from functools import lru_cache
from typing import Literal, Optional, Union

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

from .retnet import MultiScaleRetention, RetNetRelPos


class RetentionLayer(nn.Module):
    """Single Retention layer with feed-forward network.
    
    This is similar to a Transformer layer but uses Retention instead of attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "swish",
    ):
        super().__init__()
        
        self.retention = MultiScaleRetention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            value_factor=2,
            gate_fn=activation,
        )
        
        self.retention_layer_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU() if activation == "gelu" else nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor], rel_pos) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual mechanism.
        
        Args:
            hidden_states: (batch, seq_len, embed_dim) - Input hidden states
            residual: (batch, seq_len, embed_dim) or None - Accumulated residual from previous layers
            rel_pos: Relative position from RetNetRelPos
            
        Returns:
            tuple of (hidden_states, residual): Both (batch, seq_len, embed_dim)
        """
        # Add residual and normalize
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.retention_layer_norm(residual)
        
        # Retention
        hidden_states = self.retention(hidden_states, rel_pos=rel_pos, chunkwise_recurrent=True)
        hidden_states = self.dropout(hidden_states)
        
        # Add residual and normalize for FFN
        residual = hidden_states + residual
        hidden_states = self.ffn_layer_norm(residual)
        
        # Feed-forward
        hidden_states = self.ffn(hidden_states)
        
        return hidden_states, residual


class RetentionEncoder(nn.Module):
    """Stack of Retention layers forming an encoder, with optional bidirectional processing.
    
    Handles both forward and backward processing internally with residual accumulation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        recurrent_chunk_size: int = 500,
        bidirectional: bool = False,
        bidirectional_merging: Literal["concat", "add", "mul"] = "concat",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.recurrent_chunk_size = recurrent_chunk_size
        self.bidirectional = bidirectional
        self.bidirectional_merging = bidirectional_merging
        
        if bidirectional_merging not in ["concat", "add", "mul"]:
            raise ValueError(f"Invalid bidirectional_merging: {bidirectional_merging}")
        
        # Relative position encoding
        self.rel_pos = RetNetRelPos(
            embed_dim=embed_dim,
            num_heads=num_heads,
            recurrent_chunk_size=recurrent_chunk_size,
        )
        
        # Forward layers
        self.forward_layers = nn.ModuleList([
            RetentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Backward layers (if bidirectional)
        if self.bidirectional:
            self.backward_layers = nn.ModuleList([
                RetentionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with bidirectional processing and residual accumulation.
        
        Args:
            input: (batch, seq_len, embed_dim)
            
        Returns:
            (batch, seq_len, embed_dim) or (batch, seq_len, embed_dim*2) if bidirectional and concat
        """
        seq_len = input.size(1)
        
        # Get relative position encoding (same for both directions - the sequence flip handles directionality)
        rel_pos = self.rel_pos.forward(slen=seq_len, chunkwise_recurrent=True)
        
        # Forward branch with residual accumulation
        for_residual = None
        forward_f = input.clone()
        for layer in self.forward_layers:
            forward_f, for_residual = layer(forward_f, for_residual, rel_pos)
        # Combine final hidden states with accumulated residual
        residual = (forward_f + for_residual) if for_residual is not None else forward_f
        
        # Backward branch (if bidirectional)
        if hasattr(self, "backward_layers"):
            back_residual = None
            backward_f = torch.flip(input, [1])  # Flip sequence for backward processing
            for layer in self.backward_layers:
                # Use same rel_pos - the flipped input naturally reverses the attention direction
                backward_f, back_residual = layer(backward_f, back_residual, rel_pos)
            # Combine final hidden states with accumulated residual
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f
            
            # Flip back to align with forward
            back_residual = torch.flip(back_residual, [1])
            
            # Merge forward and backward
            if self.bidirectional_merging == "concat":
                residual = torch.cat([residual, back_residual], -1)
            elif self.bidirectional_merging == "add":
                residual += back_residual
            else:  # mul
                residual = torch.mul(residual, back_residual)
        
        return residual


class WavLMBiRetention(BaseModel):
    """WavLM + Bidirectional Retention model for speaker diarization.
    
    Architecture:
        WavLM -> Projection -> BiRetNet -> Classifier
        
    Parameters
    ----------
    wav2vec : str or dict, optional
        WavLM model specification. Defaults to "WAVLM_BASE".
    wav2vec_layer : int, optional
        Which WavLM layer to use. -1 (default) uses weighted average of all layers.
    retention : dict, optional
        Retention encoder configuration with keys:
        - embed_dim: Hidden dimension (default: 256)
        - num_layers: Number of retention layers (default: 4)
        - num_heads: Number of attention heads (default: 4)
        - ffn_dim: Feed-forward dimension (default: 1024)
        - dropout: Dropout rate (default: 0.1)
        - recurrent_chunk_size: Chunk size for recurrent processing (default: 100)
    bidirectional : bool, optional
        Enable bidirectional processing. Defaults to True.
    bidirectional_merging : str, optional
        How to merge forward and backward outputs:
        - "concat": Concatenate (doubles feature dimension)
        - "add": Element-wise addition (keeps dimension)
        - "mul": Element-wise multiplication (keeps dimension)
        Defaults to "add".
    freeze_wavlm : bool, optional
        Whether to freeze WavLM parameters. Defaults to True.
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    RETENTION_DEFAULTS = {
        "embed_dim": 256,
        "num_stacks": 4,
        "num_layers": 1,
        "num_heads": 4,
        "ffn_dim": 1024,
        "dropout": 0.1,
        "recurrent_chunk_size": 100,
    }

    def __init__(
        self,
        wav2vec: Union[dict, str] = None,
        wav2vec_layer: int = -1,
        retention: Optional[dict] = None,
        bidirectional: bool = True,
        bidirectional_merging: Literal["concat", "add", "mul"] = "add",
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Task = None,
        freeze_wavlm: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        
        self.freeze_wavlm = freeze_wavlm
        self.bidirectional = bidirectional
        self.bidirectional_merging = bidirectional_merging
        
        if bidirectional_merging not in ["concat", "add", "mul"]:
            raise ValueError(f"Invalid bidirectional_merging: {bidirectional_merging}")
        
        if wav2vec is None:
            wav2vec = self.WAV2VEC_DEFAULTS
        
        if isinstance(wav2vec, str):
            # Load from torchaudio pipelines
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle._sample_rate:
                    raise ValueError(f"Expected {bundle._sample_rate}Hz, found {sample_rate}Hz.")
                wav2vec_dim = bundle._params["encoder_embed_dim"]
                wav2vec_num_layers = bundle._params["encoder_num_layers"]
                self.wav2vec = bundle.get_model()
                if hasattr(self.wav2vec, "model"):
                    self.wav2vec = self.wav2vec.model
            else:
                # Load from checkpoint
                _checkpoint = torch.load(wav2vec)
                wav2vec = _checkpoint.pop("config")
                self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
                state_dict = _checkpoint.pop("state_dict")
                self.wav2vec.load_state_dict(state_dict)
                wav2vec_dim = wav2vec["encoder_embed_dim"]
                wav2vec_num_layers = wav2vec["encoder_num_layers"]
        elif isinstance(wav2vec, dict):
            self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
            wav2vec_dim = wav2vec["encoder_embed_dim"]
            wav2vec_num_layers = wav2vec["encoder_num_layers"]

        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(data=torch.ones(wav2vec_num_layers), requires_grad=True)

        retention = merge_dict(self.RETENTION_DEFAULTS, retention)
        
        self.save_hyperparameters("wav2vec", "wav2vec_layer", "retention", "bidirectional", 
                                 "bidirectional_merging", "freeze_wavlm")

        self.selected_channel = 0

        # Project WavLM features to Retention input dimension
        self.proj = nn.Linear(wav2vec_dim, self.hparams.retention["embed_dim"])
        self.lnorm = nn.LayerNorm(self.hparams.retention["embed_dim"])

        # Create multiple Retention encoder stacks
        num_stacks = self.hparams.retention.get("num_stacks", 1)
        self.retention_encoders = nn.ModuleList([
            RetentionEncoder(
                embed_dim=self.hparams.retention["embed_dim"],
                num_layers=self.hparams.retention["num_layers"],
                num_heads=self.hparams.retention["num_heads"],
                ffn_dim=self.hparams.retention["ffn_dim"],
                dropout=self.hparams.retention["dropout"],
                recurrent_chunk_size=self.hparams.retention["recurrent_chunk_size"],
                bidirectional=self.bidirectional,
                bidirectional_merging=self.bidirectional_merging,
            )
            for _ in range(num_stacks)
        ])
        
        # Freeze WavLM parameters if requested
        if self.freeze_wavlm:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

    def build(self):
        """Build classifier after task is set."""
        # Determine input dimension based on bidirectional merging
        if self.bidirectional and self.bidirectional_merging == "concat":
            classifier_input_dim = self.hparams.retention["embed_dim"] * 2
        else:
            classifier_input_dim = self.hparams.retention["embed_dim"]
        
        self.classifier = nn.Linear(classifier_input_dim, self.dimension)
        self.activation = self.default_activation()

    @property
    def dimension(self) -> int:
        """Dimension of output."""
        if isinstance(self.specifications, tuple):
            raise ValueError("WavLMBiRetention does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames.

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """
        num_frames = num_samples
        for conv_layer in self.wav2vec.feature_extractor.conv_layers:
            num_frames = conv1d_num_frames(
                num_frames,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field.

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        receptive_field_size = num_frames
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field.

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """
        receptive_field_center = frame
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_center = conv1d_receptive_field_center(
                receptive_field_center,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_center

    def on_train_epoch_end(self):
        """Log learning rate at the end of each training epoch."""
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True, prog_bar=True, logger=True)
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        assert waveforms.dim() == 3
        waveforms = waveforms[:, self.selected_channel, :]

        # Extract WavLM features
        wav2vec_num_layers = None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        
        if self.freeze_wavlm:
            with torch.no_grad():
                outputs, _ = self.wav2vec.extract_features(waveforms, num_layers=wav2vec_num_layers)
        else:
            outputs, _ = self.wav2vec.extract_features(waveforms, num_layers=wav2vec_num_layers)

        if wav2vec_num_layers is None:
            # Combine all layers with learnable weights
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(self.wav2vec_weights, dim=0)
        else:
            # Use specific layer
            outputs = outputs[-1]

        # Project and normalize
        outputs = self.proj(outputs)
        outputs = self.lnorm(outputs)
        
        # Store original length
        original_frames = outputs.size(1)
        
        # Pad to multiples of recurrent_chunk_size
        import math
        chunk_size = self.hparams.retention["recurrent_chunk_size"]
        padded_frames = math.ceil(original_frames / chunk_size) * chunk_size
        if padded_frames > original_frames:
            padding_needed = padded_frames - original_frames
            last_frame = outputs[:, -1:, :]
            padding = last_frame.repeat(1, padding_needed, 1)
            outputs = torch.cat([outputs, padding], dim=1)
        
        # Process through multiple Retention encoder stacks
        for retention_encoder in self.retention_encoders:
            outputs = retention_encoder(outputs)
        
        # Crop back to original length
        outputs = outputs[:, :original_frames, :]

        # Classifier
        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)

        return outputs
