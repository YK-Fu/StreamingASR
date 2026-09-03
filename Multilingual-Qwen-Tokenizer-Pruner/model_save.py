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


def update_token_id_in_config(config_dict, key, token_mapping, new_id_by_old=None):
    """Safely update a token ID in config using the new mapping."""
    if new_id_by_old is None:
        new_id_by_old = {old_id: new_id for new_id, old_id in enumerate(token_mapping)}
    if key in config_dict:
        old_id = config_dict[key]
        if isinstance(old_id, int) and old_id in new_id_by_old:
            config_dict[key] = new_id_by_old[old_id]
        elif isinstance(old_id, list):
            config_dict[key] = [new_id_by_old.get(t, t) for t in old_id]


def saving_updated_qwenvl(old_model, new_vocab_size, token_mapping, output_path,
                          num_extra_special=0, merge_init=None, num_mapped_bpe=None,
                          num_padding_special=0):
    """Save updated Qwen-VL model with new vocabulary.

    Row layout matches Hugging Face tokenizers:
    [mapped BPE][merged-char BPE][retained special][extra special].
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

    new_id_by_old = _populate_vocab_rows(
        new_embeds, new_lm_head, embed_layer, lm_head, old_model,
        token_mapping, new_vocab_size, num_extra_special, merge_init,
        tied, num_mapped_bpe, num_padding_special,
    )

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
    update_token_id_in_config(old_model.config.__dict__, 'eos_token_id', token_mapping, new_id_by_old)
    update_token_id_in_config(old_model.config.__dict__, 'pad_token_id', token_mapping, new_id_by_old)
    update_token_id_in_config(old_model.config.__dict__, 'bos_token_id', token_mapping, new_id_by_old)
    
    if 'visual' in old_model.config.__dict__ and "image_start_id" in old_model.config.__dict__['visual']:
        old_id = old_model.config.__dict__['visual']["image_start_id"]
        old_model.config.__dict__['visual']["image_start_id"] = new_id_by_old[old_id]
    
    # Update generation config
    if hasattr(old_model, 'generation_config'):
        update_token_id_in_config(old_model.generation_config.__dict__, 'eos_token_id', token_mapping, new_id_by_old)
        update_token_id_in_config(old_model.generation_config.__dict__, 'pad_token_id', token_mapping, new_id_by_old)
        update_token_id_in_config(old_model.generation_config.__dict__, 'bos_token_id', token_mapping, new_id_by_old)
    
    # Save new model
    print(f"Saving new model checkpoint to {output_path}")
    old_model.save_pretrained(output_path)


def _populate_vocab_rows(new_embeds, new_lm_head, embed_layer, lm_head, old_model,
                         token_mapping, new_vocab_size, num_extra_special,
                         merge_init, tied, num_mapped_bpe=None,
                         num_padding_special=0):
    """Populate rows in HF-compatible order with all special tokens at the tail.

    Layout: [mapped BPE][merged-char BPE][retained special][extra special].
    ``token_mapping`` contains mapped BPE IDs followed by retained source-special
    IDs. Newly merged BPE rows have no source ID and are mean-initialized; new
    user-provided special rows are initialized from the source EOS row. Reserved
    vocabulary-padding rows are the final rows and are initialized to zero.
    """
    def set_row(idx, emb_val, lm_val):
        new_embeds.weight.data[idx] = emb_val
        if not tied:
            new_lm_head.weight.data[idx] = lm_val

    num_mapped = len(token_mapping)
    if num_mapped_bpe is None:
        num_mapped_bpe = num_mapped
    assert 0 <= num_mapped_bpe <= num_mapped
    assert len(set(token_mapping)) == num_mapped
    if not 0 <= num_padding_special <= num_extra_special:
        raise ValueError(
            f"num_padding_special must be in [0, {num_extra_special}], "
            f"got {num_padding_special}"
        )

    mapped_bpe = token_mapping[:num_mapped_bpe]
    mapped_special = token_mapping[num_mapped_bpe:]
    if mapped_bpe:
        idx = torch.LongTensor(mapped_bpe).to(old_model.device)
        new_embeds.weight.data[:num_mapped_bpe] = embed_layer.weight.data[idx]
        if not tied:
            new_lm_head.weight.data[:num_mapped_bpe] = lm_head.weight.data[idx]

    if merge_init:
        base = num_mapped_bpe
        for j, ids in enumerate(merge_init):
            idx = torch.LongTensor(ids).to(old_model.device)
            set_row(base + j,
                    embed_layer.weight.data[idx].mean(dim=0),
                    lm_head.weight.data[idx].mean(dim=0))
        print(f"  {len(merge_init)} merged single-char token(s) initialized from the "
              f"mean of their original sub-token embeddings")

    special_base = num_mapped_bpe + len(merge_init or [])
    if mapped_special:
        idx = torch.LongTensor(mapped_special).to(old_model.device)
        end = special_base + len(mapped_special)
        new_embeds.weight.data[special_base:end] = embed_layer.weight.data[idx]
        if not tied:
            new_lm_head.weight.data[special_base:end] = lm_head.weight.data[idx]

    if num_extra_special > 0:
        proto_id = old_model.config.eos_token_id
        extra_base = special_base + len(mapped_special)
        num_user_extra_special = num_extra_special - num_padding_special
        for j in range(num_user_extra_special):
            set_row(extra_base + j,
                    embed_layer.weight.data[proto_id],
                    lm_head.weight.data[proto_id])
        if num_user_extra_special:
            print(f"  {num_user_extra_special} extra special token(s) initialized from "
                  f"source token id {proto_id} (<|endoftext|>)")
        if num_padding_special:
            padding_base = extra_base + num_user_extra_special
            new_embeds.weight.data[padding_base:].zero_()
            if not tied:
                new_lm_head.weight.data[padding_base:].zero_()
            print(f"  {num_padding_special} vocabulary-padding row(s) zero-initialized")

    expected = num_mapped + num_extra_special + len(merge_init or [])
    assert expected == new_vocab_size, \
        f"vocabulary row layout mismatch: {expected} != new_vocab_size {new_vocab_size}"

    new_id_by_old = {old_id: new_id for new_id, old_id in enumerate(mapped_bpe)}
    new_id_by_old.update({
        old_id: special_base + j for j, old_id in enumerate(mapped_special)
    })
    return new_id_by_old


def saving_updated_qwen(old_model, new_vocab_size, token_mapping, output_path,
                        num_extra_special=0, merge_init=None, num_mapped_bpe=None,
                        num_padding_special=0):
    """Save updated Qwen/Qwen2 model with new vocabulary.

    Rows are laid out as [mapped BPE][merged-char BPE][retained special]
    [extra special]; see _populate_vocab_rows.
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

    new_id_by_old = _populate_vocab_rows(
        new_embeds, new_lm_head, embed_layer, lm_head, old_model,
        token_mapping, new_vocab_size, num_extra_special, merge_init,
        tied, num_mapped_bpe, num_padding_special,
    )

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
    update_token_id_in_config(old_model.config.__dict__, 'eos_token_id', token_mapping, new_id_by_old)
    update_token_id_in_config(old_model.config.__dict__, 'pad_token_id', token_mapping, new_id_by_old)
    update_token_id_in_config(old_model.config.__dict__, 'bos_token_id', token_mapping, new_id_by_old)
    
    # Update generation config token IDs
    if hasattr(old_model, 'generation_config'):
        gen_config = old_model.generation_config.__dict__
        update_token_id_in_config(gen_config, 'eos_token_id', token_mapping, new_id_by_old)
        update_token_id_in_config(gen_config, 'pad_token_id', token_mapping, new_id_by_old)
        update_token_id_in_config(gen_config, 'bos_token_id', token_mapping, new_id_by_old)
    
    # Save new model
    print(f"Saving new model checkpoint to {output_path}")
    old_model.save_pretrained(output_path)
