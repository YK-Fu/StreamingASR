
import torch
import torch.nn as nn
from transformers import AutoModel
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
        self.prediction = AutoModel.from_config(self.config)
        # Enable gradient checkpointing if specified in config
        if gradient_checkpointing:
            logging.info("Enabling gradient checkpointing for decoder")
            self.prediction.gradient_checkpointing_enable()
        if freeze:
            for name, param in self.named_parameters():
                if 'embed' in name and config.tie_word_embeddings:
                    continue
                param.requires_grad = False
        elif freeze_ffn:
            for name, param in self.named_parameters():
                if '.mlp.' in name:
                    param.requires_grad = False

    def forward(self, input_ids, position_ids, attn_mask=None, cache=None, cache_position=None):
        output = self.prediction(
            input_ids=input_ids, 
            position_ids=position_ids,
            attention_mask=attn_mask, 
            past_key_values=cache, 
            cache_position=cache_position,
            use_cache=not self.training
        )
        g, states = output.last_hidden_state, output.past_key_values  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, states
