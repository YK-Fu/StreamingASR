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


def saving_updated_qwenvl(old_model, new_vocab_size, token_mapping, output_path,
                          num_extra_special=0, merge_init=None):
    """Save updated Qwen-VL model with new vocabulary.

    Appended-row layout matches saving_updated_qwen: [extra_special][merged_char].
    """
    embed_layer, lm_head, model_type = get_embed_and_lm_head(old_model)
    tied = bool(getattr(old_model.config, 'tie_word_embeddings', False))

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

    num_mapped = len(token_mapping)
    assert len(set(token_mapping)) == num_mapped
    mapping_tensor = torch.LongTensor(token_mapping).to(old_model.device)
    new_embeds.weight.data[:num_mapped] = embed_layer.weight.data[mapping_tensor]
    if not tied:
        new_lm_head.weight.data[:num_mapped] = lm_head.weight.data[mapping_tensor]

    _init_appended_rows(new_embeds, new_lm_head, embed_layer, lm_head, old_model,
                        num_mapped, new_vocab_size, num_extra_special, merge_init, tied)

    if tied:
        new_lm_head.weight = new_embeds.weight

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


def _init_appended_rows(new_embeds, new_lm_head, embed_layer, lm_head, old_model,
                        num_mapped, new_vocab_size, num_extra_special, merge_init, tied):
    """Initialize the appended rows: [extra_special][merged_char].

    Layout after the num_mapped mapped rows:
      - num_extra_special rows  -> seeded from the source <|endoftext|> row
        (default Embedding init is N(0,1), ~60x larger than pretrained rows; a sane
        direction matters at every BOS/lang-id position).
      - len(merge_init) rows    -> mean of the original-Qwen sub-token rows of the
        character, so each merged single-char token starts as the average of the
        fragments it replaces (a standard subword-mean warm start, refined by training).
    """
    def set_row(idx, emb_val, lm_val):
        new_embeds.weight.data[idx] = emb_val
        if not tied:
            new_lm_head.weight.data[idx] = lm_val

    if num_extra_special > 0:
        proto_id = old_model.config.eos_token_id
        for j in range(num_extra_special):
            set_row(num_mapped + j,
                    embed_layer.weight.data[proto_id],
                    lm_head.weight.data[proto_id])
        print(f"  {num_extra_special} extra special token(s) initialized from "
              f"source token id {proto_id} (<|endoftext|>)")

    if merge_init:
        base = num_mapped + num_extra_special
        for j, ids in enumerate(merge_init):
            idx = torch.LongTensor(ids).to(old_model.device)
            set_row(base + j,
                    embed_layer.weight.data[idx].mean(dim=0),
                    lm_head.weight.data[idx].mean(dim=0))
        print(f"  {len(merge_init)} merged single-char token(s) initialized from the "
              f"mean of their original sub-token embeddings")

    expected = num_mapped + num_extra_special + len(merge_init or [])
    assert expected == new_vocab_size, \
        f"appended-row layout mismatch: {expected} != new_vocab_size {new_vocab_size}"


def saving_updated_qwen(old_model, new_vocab_size, token_mapping, output_path,
                        num_extra_special=0, merge_init=None):
    """Save updated Qwen/Qwen2 model with new vocabulary.

    Appended rows beyond the mapped vocab are laid out as
    [extra_special tokens][merged single-char tokens]; see _init_appended_rows.
    Respects tie_word_embeddings (input/output share one weight when tied).
    """
    embed_layer, lm_head, model_type = get_embed_and_lm_head(old_model)
    tied = bool(getattr(old_model.config, 'tie_word_embeddings', False))

    print(f"Detected model architecture: {model_type} (tie_word_embeddings={tied})")

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

    num_mapped = len(token_mapping)
    assert len(set(token_mapping)) == num_mapped, \
        f"Mapping has duplicates: {len(set(token_mapping))} unique vs {num_mapped} expected"

    mapping_tensor = torch.LongTensor(token_mapping).to(old_model.device)
    new_embeds.weight.data[:num_mapped] = embed_layer.weight.data[mapping_tensor]
    if not tied:
        new_lm_head.weight.data[:num_mapped] = lm_head.weight.data[mapping_tensor]

    _init_appended_rows(new_embeds, new_lm_head, embed_layer, lm_head, old_model,
                        num_mapped, new_vocab_size, num_extra_special, merge_init, tied)

    # When tied, input and output share one weight tensor (lm_head is retied below).
    if tied:
        new_lm_head.weight = new_embeds.weight

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
