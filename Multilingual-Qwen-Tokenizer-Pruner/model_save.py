"""
Functions for saving model checkpoints with updated vocabulary.
"""

import os
import torch


def get_embed_and_lm_head(model):
    """
    Get embedding layer and lm_head for different model architectures.
    Returns: (embed_layer, lm_head, model_type)
    """
    # Qwen2 / Qwen2.5 architecture (HuggingFace format)
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens, model.lm_head, 'qwen2'
    # Original Qwen architecture
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte, model.lm_head, 'qwen1'
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def update_token_id_in_config(config_dict, key, token_mapping):
    """Safely update a token ID in config using the new mapping."""
    if key in config_dict:
        old_id = config_dict[key]
        if isinstance(old_id, int) and old_id in token_mapping:
            config_dict[key] = token_mapping.index(old_id)
        elif isinstance(old_id, list):
            config_dict[key] = [token_mapping.index(t) if t in token_mapping else t for t in old_id]


def saving_updated_qwenvl(old_model, new_vocab_size, token_mapping, output_path):
    """Save updated Qwen-VL model with new vocabulary."""
    embed_layer, lm_head, model_type = get_embed_and_lm_head(old_model)
    
    # Define new modules
    new_embeds = torch.nn.Embedding(
        new_vocab_size, 
        old_model.config.hidden_size, 
        dtype=embed_layer.weight.dtype
    )
    new_lm_head = torch.nn.Linear(
        old_model.config.hidden_size, 
        new_vocab_size, 
        bias=False, 
        dtype=lm_head.weight.dtype
    )
    
    # Get new module parameters from the old
    assert len(set(token_mapping)) == new_vocab_size
    mapping_tensor = torch.LongTensor(token_mapping).to(old_model.device)
    new_embeds.weight.data = embed_layer.weight.data[mapping_tensor]
    new_lm_head.weight.data = lm_head.weight.data[mapping_tensor]
    
    # Update model weights
    if model_type == 'qwen2':
        old_model.model.embed_tokens.weight = new_embeds.weight
        old_model.model.embed_tokens.num_embeddings = new_vocab_size
    else:
        old_model.transformer.wte.weight = new_embeds.weight
        old_model.transformer.wte.num_embeddings = new_vocab_size
    
    old_model.lm_head.weight = new_lm_head.weight
    old_model.lm_head.out_features = new_vocab_size
    
    # Update config
    old_model.config.__dict__['vocab_size'] = new_vocab_size
    old_model.config.__dict__['_name_or_path'] = output_path
    
    if 'visual' in old_model.config.__dict__ and "image_start_id" in old_model.config.__dict__['visual']:
        old_model.config.__dict__['visual']["image_start_id"] = token_mapping.index(
            old_model.config.__dict__['visual']["image_start_id"])
    
    # Update generation config
    if hasattr(old_model, 'generation_config'):
        update_token_id_in_config(old_model.generation_config.__dict__, 'eos_token_id', token_mapping)
        update_token_id_in_config(old_model.generation_config.__dict__, 'pad_token_id', token_mapping)
        update_token_id_in_config(old_model.generation_config.__dict__, 'bos_token_id', token_mapping)
    
    # Save new model
    print(f"Saving new model checkpoint to {output_path}")
    old_model.save_pretrained(output_path)


def saving_updated_qwen(old_model, new_vocab_size, token_mapping, output_path):
    """Save updated Qwen/Qwen2 model with new vocabulary."""
    embed_layer, lm_head, model_type = get_embed_and_lm_head(old_model)
    
    print(f"Detected model architecture: {model_type}")
    
    # Define new modules
    new_embeds = torch.nn.Embedding(
        new_vocab_size, 
        old_model.config.hidden_size, 
        dtype=embed_layer.weight.dtype
    )
    new_lm_head = torch.nn.Linear(
        old_model.config.hidden_size, 
        new_vocab_size, 
        bias=False, 
        dtype=lm_head.weight.dtype
    )
    
    # Get new module parameters from the old
    assert len(set(token_mapping)) == new_vocab_size, \
        f"Mapping has duplicates: {len(set(token_mapping))} unique vs {new_vocab_size} expected"
    
    mapping_tensor = torch.LongTensor(token_mapping).to(old_model.device)
    new_embeds.weight.data = embed_layer.weight.data[mapping_tensor]
    new_lm_head.weight.data = lm_head.weight.data[mapping_tensor]
    
    # Update model weights based on architecture
    if model_type == 'qwen2':
        # Qwen2 / Qwen2.5 architecture
        old_model.model.embed_tokens.weight = new_embeds.weight
        old_model.model.embed_tokens.num_embeddings = new_vocab_size
    else:
        # Original Qwen architecture
        old_model.transformer.wte.weight = new_embeds.weight
        old_model.transformer.wte.num_embeddings = new_vocab_size
    
    old_model.lm_head.weight = new_lm_head.weight
    old_model.lm_head.out_features = new_vocab_size
    
    # Update config
    old_model.config.__dict__['vocab_size'] = new_vocab_size
    old_model.config.__dict__['_name_or_path'] = output_path
    
    # Update generation config token IDs
    if hasattr(old_model, 'generation_config'):
        gen_config = old_model.generation_config.__dict__
        update_token_id_in_config(gen_config, 'eos_token_id', token_mapping)
        update_token_id_in_config(gen_config, 'pad_token_id', token_mapping)
        update_token_id_in_config(gen_config, 'bos_token_id', token_mapping)
    
    # Save new model
    print(f"Saving new model checkpoint to {output_path}")
    old_model.save_pretrained(output_path)
