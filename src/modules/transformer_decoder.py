
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from nemo.utils import logging

dtype_map = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

class LLMDecoder(nn.Module):
    def __init__(self, config, gradient_checkpointing=False, dtype="bf16", freeze=False, freeze_ffn=False, **kwargs):
        super().__init__()
        config.torch_dtype = dtype_map[dtype]
        self.config = config
        self.prediction = AutoModelForCausalLM.from_config(self.config)
        # Enable gradient checkpointing if specified in config
        if gradient_checkpointing:
            logging.info("Enabling gradient checkpointing for decoder")
            self.prediction.gradient_checkpointing_enable()
        if freeze:
            for name, param in self.named_parameters():
                if 'embed' in name or 'lm_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif freeze_ffn:
            for name, param in self.named_parameters():
                if '.mlp.' in name:
                    param.requires_grad = False

    def forward(
        self,
        input_ids,
        position_ids=None,
        attn_mask=None,
        cache=None,
        cache_position=None,
        return_lm_logits=False,
        output_indices=None,
    ):
        output = self.prediction.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attn_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=cache is not None
        )

        h = output.last_hidden_state          # (B, U, D)
        if output_indices is not None:
            if output_indices.ndim != 2 or output_indices.shape[0] != h.shape[0]:
                raise ValueError("output_indices must have shape [batch, output_time]")
            h = h.gather(
                1, output_indices.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            )
        states = output.past_key_values
        g = h.transpose(1, 2)                 # (B, D, U)
        if return_lm_logits:
            lm_logits = self.prediction.lm_head(h)   # (B, U, V)
            return g, lm_logits, states
        return g, states
