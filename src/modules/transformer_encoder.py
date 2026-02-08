# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Modifications Copyright (c) 2026, Iven Fu. All rights reserved.

from collections import OrderedDict

import torch
import torch.nn as nn

from nemo.core.classes.common import typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    StringType,
    NeuralType,
    SpectrogramType,
)

from src.modules.transformer_layer import TransformerLayer, Cache

__all__ = ['WhisperEncoder']


class WhisperEncoder(NeuralModule, AccessMixin):
    def input_example(self, max_batch=1, max_dim=256, mode='asr'):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
        all_input_example = tuple([input_example, mode])

        return all_input_example

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "mode": NeuralType(tuple(), StringType(), optional=True),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            }
        )


    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        ff_expansion_factor=4,
        n_heads=4,
        pos_emb_max_len=5000,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        gradient_checkpointing: bool = False,
        freeze: bool = False,
        freeze_ffn: bool = False,
        position_embedding_type: str = "alibi",
    ):
        """
        Args:
            feat_in (int): the size of feature channels
            n_layers (int): number of layers of TransformerLayer
            d_model (int): the hidden size of the model
            ff_expansion_factor (int): the expansion factor in feed forward layers
            n_heads (int): number of heads in multi-headed attention layers
            pos_emb_max_len (int): the maximum length of positional embeddings (for learned positional embeddings)
            use_bias (bool): Use bias in all Linear and Conv1d layers from each ConformerLayer to improve
            activation flow and stabilize training of huge models.
            Defaults to True.
            dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
            dropout_pre_encoder (float): the dropout rate used before the encoder
            Defaults to 0.1.
            dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
            dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
            gradient_checkpointing (bool): whether to use gradient checkpointing for the TransformerLayer
            Defaults to False.
            freeze (bool): whether to freeze the encoder
            Defaults to False.
            freeze_ffn (bool): whether to freeze the ffn network in each of the transformer blocks. This reduces the memory usage of the model.
            Defaults to False.
            position_embedding_type (str): "alibi" or "learned"
            Defaults to "alibi".
        """
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.subsampling_factor = 2

        self.gradient_checkpointing = gradient_checkpointing
        # Subsampling
        self.pre_encode = nn.Sequential(
            nn.Conv1d(feat_in, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Dropout(dropout_pre_encoder),
        )
        self.max_audio_length = pos_emb_max_len
        # Positional encodings
        if position_embedding_type == "learned":
            self.pos_enc = nn.Embedding(self.max_audio_length, d_model)
            self.pos_dropout = nn.Dropout(dropout_emb)
        else:
            self.pos_enc = None

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = TransformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                attn_type="alibi" if position_embedding_type == "alibi" else "full",
                dropout=dropout,
                dropout_att=dropout_att,
                dropout_emb=dropout_emb,
                use_bias=use_bias,
                gradient_checkpointing=self.gradient_checkpointing,
            )
            self.layers.append(layer)
        self.layer_norm = nn.LayerNorm(d_model)

        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        elif freeze_ffn:
            for name, param in self.named_parameters():
                if '.feed_forward.' in name:
                    param.requires_grad = False

    @typecheck()
    # @torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
    def forward(
        self,
        audio_signal,
        cache=None,
        mode='asr',
    ):
        """
        Forward function for the TransformerEncoder accepting an audio signal.
        mode: 'asr' or 'teacher'
            - 'asr': ASR mode, the encoder is used for ASR training
            - 'teacher': Distillation mode, the encoder is used for distilling the Whisper model
        """
        
        if mode == 'asr':
            return self.forward_internal(audio_signal, cache=cache)
        elif mode == 'teacher':
            return self.forward_teacher(audio_signal)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @torch.no_grad()
    def forward_teacher(self, audio_signal):
        audio_signal = self.pre_encode(audio_signal)
        audio_signal = torch.transpose(audio_signal, 1, 2)      # (B, D, T) -> (B, T, D)
        position_ids = torch.arange(0, audio_signal.size(1), device=audio_signal.device)
        pos_emb = self.pos_enc(position_ids)
        audio_signal = audio_signal + pos_emb
        for layer in self.layers:
            audio_signal = layer(x=audio_signal)
        audio_signal = self.layer_norm(audio_signal)
        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal

    def forward_internal(
        self,
        audio_signal,
        cache=None,
    ):
        audio_signal = self.pre_encode(audio_signal)
        audio_signal = audio_signal.transpose(1, 2)     # (B, D, T) -> (B, T, D)

        if self.pos_enc is not None:
            position_ids = torch.arange(0, audio_signal.size(1), device=audio_signal.device)
            pos_emb = self.pos_enc(position_ids)
            audio_signal = audio_signal + self.pos_dropout(pos_emb)

        for lth, layer in enumerate(self.layers):
            if cache is not None:
                cache_lth = cache[lth]
            else:
                cache_lth = None
            audio_signal = layer(x=audio_signal, cache=cache_lth)

            # saving tensors if required for interctc loss
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )

        audio_signal = self.layer_norm(audio_signal)
        audio_signal = torch.transpose(audio_signal, 1, 2)

        return audio_signal

    def get_initial_cache_state(self, batch_size=1, max_cache_len=1500, dtype=torch.bfloat16, device=None):
        if device is None:
            device = next(self.parameters()).device
        cache = [Cache(
            max_cache_len=max_cache_len,
            batch_size=batch_size,
            num_heads=self.n_heads,
            head_dim=self.d_model // self.n_heads,
            device=device,
            dtype=dtype,
        ) for _ in range(self.n_layers)]
        return cache
