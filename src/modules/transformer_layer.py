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

import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from dataclasses import dataclass, field
from nemo.core.classes.mixins import AccessMixin

__all__ = ['TransformerLayer']


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes following the original paper.
    
    For power-of-2 heads: slopes = 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    For non-power-of-2 heads: interpolate between closest power-of-2 slopes
    
    Reference: https://arxiv.org/abs/2108.12409
    """
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    
    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    else:
        # For non-power-of-2, get slopes from closest power-of-2 and interpolate
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        # Get extra slopes from double the closest power-of-2, taking every other one
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
        return torch.tensor(slopes + extra_slopes)

@dataclass
class Cache:
    max_cache_len: int
    batch_size: int
    num_heads: int
    head_dim: int

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    chunk_start_ptr: int = field(init=False)
    chunk_end_ptr: int = field(init=False)
    full: bool = field(init=False)
    k: torch.Tensor = field(init=False)
    v: torch.Tensor = field(init=False)
    

    def __post_init__(self):
        shape = (self.batch_size, self.num_heads, self.max_cache_len, self.head_dim)
        self.k = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.chunk_start_ptr = 0
        self.chunk_end_ptr = 0
        self.full = False

    def reset(self):
        self.chunk_start_ptr = 0
        self.chunk_end_ptr = 0
        self.full = False

    def update(self, k, v):
        assert k.shape[2] == v.shape[2], "k and v must have the same length"
        assert k.shape[2] <= self.max_cache_len, "k must be shorter than or equal to max_cache_len"
        self.chunk_start_ptr = self.chunk_end_ptr
        self.chunk_end_ptr = (self.chunk_start_ptr + k.shape[2]) % self.max_cache_len
        if self.chunk_start_ptr < self.chunk_end_ptr:
            self.k[:, :, self.chunk_start_ptr:self.chunk_end_ptr] = k
            self.v[:, :, self.chunk_start_ptr:self.chunk_end_ptr] = v
        else:
            # Rolling overwrite to the beginning of the cache
            self.full = True
            self.k[:, :, self.chunk_start_ptr:] = k[:, :, :self.max_cache_len - self.chunk_start_ptr]
            self.v[:, :, self.chunk_start_ptr:] = v[:, :, :self.max_cache_len - self.chunk_start_ptr]
            self.k[:, :, :self.chunk_end_ptr] = k[:, :, self.max_cache_len - self.chunk_start_ptr:]
            self.v[:, :, :self.chunk_end_ptr] = v[:, :, self.max_cache_len - self.chunk_start_ptr:]

class MultiHeadAttention(nn.Module):

    def __init__(self, n_feat, n_head, dropout_rate, dropout_emb, use_bias=True, attn_type="alibi"):

        super(MultiHeadAttention, self).__init__()
        self.cache_drop_size = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.n_heads = n_head
        self.head_dim = n_feat // n_head
        self.linear_qkv = nn.Linear(n_feat, n_feat * 3, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout_rate = dropout_rate
        self.dropout_emb = dropout_emb
        self.attn_type = attn_type
        if attn_type == "alibi":
            # Use proper ALiBi slopes from the original paper (handles non-power-of-2 heads)
            self.register_buffer("slopes", get_alibi_slopes(self.n_heads))

    def _get_score_mask_mod(self, q, k, v, cache):
        if self.attn_type == "alibi":
            if cache is None:
                q_offset = 0
                kv_offset = 0
                max_cache_len = None
            else:
                max_cache_len = cache.max_cache_len
                if not cache.full:
                    q_offset = cache.chunk_start_ptr
                    kv_offset = 0
                else:
                    q_length = abs(cache.chunk_end_ptr - cache.chunk_start_ptr)
                    q_offset = cache.max_cache_len - q_length
                    kv_offset = - cache.chunk_end_ptr

            def alibi_score_mod(score, b, h, q_idx, kv_idx):
                slope = self.slopes[h]
                relative_q_idx = q_idx + q_offset
                relative_kv_idx = kv_idx + kv_offset if max_cache_len is None else (kv_idx + kv_offset) % max_cache_len
                distance = relative_kv_idx - relative_q_idx
                return score + slope * distance

            def alibi_mask_mod(b, h, q_idx, kv_idx):
                relative_q_idx = q_idx + q_offset
                relative_kv_idx = kv_idx + kv_offset if max_cache_len is None else (kv_idx + kv_offset) % max_cache_len
                return relative_q_idx >= relative_kv_idx
            return alibi_score_mod, create_block_mask(alibi_mask_mod, B=None, H=None, Q_LEN=q.shape[2], KV_LEN=k.shape[2] if max_cache_len is None else max_cache_len)

        elif self.attn_type == "full":
            return None, None

        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

    def forward(self, x, cache=None):
        b, s, _ = x.shape

        qkv = self.linear_qkv(x)
        qkv = qkv.view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None:
            assert self.attn_type == "alibi", "cache is only supported for alibi attention (with rolling cache)"
            cache.update(k, v)
            k = cache.k
            v = cache.v
        score_mod, block_mask = self._get_score_mask_mod(q, k, v, cache)
        attn_output = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        attn_output = attn_output.transpose(1, 2).view(b, s, self.n_heads * self.head_dim)
        attn_output = self.linear_out(attn_output)
        return attn_output

class TransformerLayer(nn.Module, AccessMixin):
    """A single block of the Transformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        use_bias (bool): Apply bias to all Linear layers from each TransformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        n_heads=4,
        attn_type="alibi",
        dropout=0.1,
        dropout_att=0.1,
        dropout_emb=0.1,
        use_bias=True,
        gradient_checkpointing=False,
    ):
        super(TransformerLayer, self).__init__()

        self.n_heads = n_heads
        self.gradient_checkpointing = gradient_checkpointing
        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout_rate=dropout_att,
            dropout_emb=dropout_emb,
            use_bias=use_bias,
            attn_type=attn_type,
        )

        # second feed forward module
        self.norm_feed_forward = LayerNorm(d_model)
        self.feed_forward = TransformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cache=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            cache (Cache) : cache for MHA layers (only supported for alibi causal attention)
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_self_att(residual)
        if self.gradient_checkpointing and self.training:
            def custom_forward(x, cache):
                return self.self_attn(x, cache)
            x = torch.utils.checkpoint.checkpoint(custom_forward, x, cache)
        else:
            x = self.self_attn(x, cache)

        residual = residual + self.dropout(x)

        x = self.norm_feed_forward(residual)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)
        return x

class TransformerFeedForward(nn.Module):
    """
    feed-forward module of Transformer model.
    use_bias (bool): Apply bias to all Linear layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation=nn.GELU(), use_bias=True):
        super(TransformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)
