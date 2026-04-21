"""
Functions for saving vocabulary in tiktoken and HuggingFace formats.
"""

import os
import json
import torch
import base64
from tqdm import tqdm


def filter_long_tokens(vocab_counts, recur_counts, old_bytes_list, max_length=10):
    """
    Zero out tokens that have more than max_length characters/bytes.
    
    Args:
        vocab_counts: Token counts (will be modified in place)
        recur_counts: Recursive counts (will be modified in place)
        old_bytes_list: List of byte sequences for each token
        max_length: Maximum allowed token length in bytes (default 10)
    
    Returns:
        tuple: (vocab_counts, recur_counts, num_filtered)
    """
    num_filtered = 0
    for i in tqdm(range(len(old_bytes_list)), desc=f"Filtering tokens > {max_length} chars"):
        token_bytes = old_bytes_list[i]
        # Check byte length (approximate character length)
        if len(token_bytes) > max_length:
            if vocab_counts[i] > 0 or recur_counts[i] > 0:
                vocab_counts[i] = 0
                recur_counts[i] = 0
                num_filtered += 1
    
    print(f"Filtered {num_filtered:,} tokens with length > {max_length}")
    return vocab_counts, recur_counts, num_filtered


def _is_cjk_character(char):
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2A700 <= cp <= 0x2CEAF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def filter_multichar_cjk_tokens(vocab_counts, recur_counts, old_bytes_list):
    """
    Zero out CJK tokens that contain more than 1 character.
    A token is considered "CJK" if every character in its decoded UTF-8 string
    falls within CJK Unicode ranges. Mixed tokens (e.g. CJK + Latin) are kept.

    Returns:
        tuple: (vocab_counts, recur_counts, num_filtered)
    """
    num_filtered = 0
    for i in tqdm(range(len(old_bytes_list)), desc="Filtering multi-char CJK tokens"):
        token_bytes = old_bytes_list[i]
        try:
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            continue

        if len(token_str) > 1 and all(_is_cjk_character(c) for c in token_str):
            if vocab_counts[i] > 0 or recur_counts[i] > 0:
                vocab_counts[i] = 0
                recur_counts[i] = 0
                num_filtered += 1

    print(f"Filtered {num_filtered:,} multi-character CJK tokens")
    return vocab_counts, recur_counts, num_filtered


def reduce_to_target_size(old_vocab_size, target_vocab_size, vocab_counts, recur_counts, old_bytes_list):
    """
    Reduce vocabulary to target size by removing tokens that can be represented by sub-tokens.
    """
    total_count_with_idx = [(vocab_counts[i] + recur_counts[i], i) for i in range(old_vocab_size)]
    sorted_count_with_idx = sorted(total_count_with_idx, key=lambda x: x[0])
    remove_count = 0
    remove_target = old_vocab_size - target_vocab_size

    for i in tqdm(range(len(sorted_count_with_idx)), desc="Reducing vocabulary"):
        token_count, token_idx = sorted_count_with_idx[i]
        if remove_count >= remove_target:
            continue
        elif token_count == 0:
            remove_count += 1
        elif len(old_bytes_list[token_idx]) > 1:
            # Check if token can be represented by sub-tokens
            token = old_bytes_list[token_idx]
            b_len = len(token)
            for j in range(1, b_len):
                if (token[:j] in old_bytes_list) and (token[j:] in old_bytes_list):
                    parta_index = old_bytes_list.index(token[:j])
                    partb_index = old_bytes_list.index(token[j:])
                    if (vocab_counts[parta_index] + recur_counts[parta_index] > 0) and \
                       (vocab_counts[partb_index] + recur_counts[partb_index] > 0):
                        vocab_counts[token_idx] = 0
                        recur_counts[token_idx] = 0
                        remove_count += 1
                        break

    if remove_count < remove_target:
        print(f"Warning: Failed to reach target size. Removed {remove_count}/{remove_target} tokens.")
                    
    return vocab_counts, recur_counts


def get_new_vocab_and_map(old_bytes_list, old_vocab_size, vocab_counts, recur_counts, 
                          old_tokenizer=None, only_essential_special_tokens=True):
    """
    Build new vocabulary and mapping from new to old token IDs.
    
    Args:
        only_essential_special_tokens: If True, only add BOS, EOS, PAD tokens.
                                       If False, add all special tokens from original.
    """
    new_bytes_list = []
    mapping_new2old = []

    for i in tqdm(range(len(old_bytes_list)), desc="Building new vocabulary"):
        if vocab_counts[i] + recur_counts[i] > 0:
            new_bytes_list.append(old_bytes_list[i])
            mapping_new2old.append(i)

    # Add special token mapping
    if only_essential_special_tokens and old_tokenizer is not None:
        # Only add BOS, EOS, PAD tokens
        essential_token_ids = set()
        
        if hasattr(old_tokenizer, 'bos_token_id') and old_tokenizer.bos_token_id is not None:
            essential_token_ids.add(old_tokenizer.bos_token_id)
            print(f"Adding BOS token: id={old_tokenizer.bos_token_id}")
        
        if hasattr(old_tokenizer, 'eos_token_id') and old_tokenizer.eos_token_id is not None:
            essential_token_ids.add(old_tokenizer.eos_token_id)
            print(f"Adding EOS token: id={old_tokenizer.eos_token_id}")
        
        if hasattr(old_tokenizer, 'pad_token_id') and old_tokenizer.pad_token_id is not None:
            essential_token_ids.add(old_tokenizer.pad_token_id)
            print(f"Adding PAD token: id={old_tokenizer.pad_token_id}")
        
        # Add only essential special tokens that aren't already in mapping
        added_special = 0
        for token_id in sorted(essential_token_ids):
            if token_id not in mapping_new2old:
                mapping_new2old.append(token_id)
                added_special += 1
                print(f"  Added special token id={token_id} at new_id={len(mapping_new2old)-1}")
        
        print(f"Added {added_special} essential special tokens (BOS/EOS/PAD)")
    else:
        # Add all special tokens (original behavior)
        num_special = old_vocab_size - len(old_bytes_list)
        print(f"Adding {num_special} special tokens")
        for i in range(len(old_bytes_list), old_vocab_size):
            mapping_new2old.append(i)

    print(f"Vocabulary size: {old_vocab_size:,} => {len(mapping_new2old):,}")

    return new_bytes_list, mapping_new2old


def save_vocab(bytes_list, token_mapping, output_path, tokenizer_format='tiktoken',
               old_tokenizer=None, extra_special_tokens=None):
    """
    Save vocabulary in the appropriate format.
    
    Args:
        bytes_list: List of byte sequences for each token
        token_mapping: Mapping from new token IDs to old token IDs
        output_path: Output directory
        tokenizer_format: 'tiktoken' or 'huggingface'
        old_tokenizer: Original HuggingFace tokenizer (needed for HuggingFace format)
        extra_special_tokens: List of new special token strings to append (e.g. ["<|zh|>", "<|en|>"])
    """
    if extra_special_tokens is None:
        extra_special_tokens = []

    token_mapping_path = os.path.join(output_path, 'token_mapping.torch')
    
    # Save tiktoken format
    if tokenizer_format == 'tiktoken':
        new_tiktoken_path = os.path.join(output_path, 'qwen.tiktoken')
        with open(new_tiktoken_path, "w", encoding="utf8") as w:
            for i, token in enumerate(bytes_list):
                line = base64.b64encode(token).decode("utf8") + " " + str(i) + "\n"
                w.write(line)
        print(f"New Tiktoken BPE file (size: {len(bytes_list)}) saved to {new_tiktoken_path}")
        if extra_special_tokens:
            special_tokens_path = os.path.join(output_path, 'extra_special_tokens.json')
            base_id = len(token_mapping)
            st_map = {tok: base_id + i for i, tok in enumerate(extra_special_tokens)}
            with open(special_tokens_path, 'w', encoding='utf-8') as f:
                json.dump(st_map, f, ensure_ascii=False, indent=2)
            print(f"Extra special tokens ({len(st_map)}) saved to {special_tokens_path}")
    
    # Save HuggingFace format if original was HuggingFace
    if tokenizer_format == 'huggingface' and old_tokenizer is not None:
        save_vocab_huggingface(bytes_list, token_mapping, output_path, old_tokenizer,
                               extra_special_tokens=extra_special_tokens)

    # Save mapping index
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
    print(f"Mapping file (new -> old token) saved: {token_mapping_path}")


def _get_byte_encoder():
    """
    Get byte encoder mapping (byte value -> unicode char).
    This is GPT-2's bytes_to_unicode() function.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_vocab_huggingface(bytes_list, token_mapping, output_path, old_tokenizer,
                           only_bos_eos=True, extra_special_tokens=None):
    """
    Save vocabulary in HuggingFace tokenizer format.
    Updates tokenizer.json with the new pruned vocabulary.
    
    Args:
        only_bos_eos: If True, only add BOS and EOS as special tokens, remove all others.
        extra_special_tokens: List of new special token strings to append after the pruned vocab.
    """
    if extra_special_tokens is None:
        extra_special_tokens = []

    byte_encoder = _get_byte_encoder()
    
    # Build new vocabulary: token_str -> token_id (BPE tokens only)
    new_vocab = {}
    for i, token_bytes in enumerate(bytes_list):
        # Convert bytes to unicode string using byte encoder
        try:
            token_str = ''.join(byte_encoder.get(b, chr(b)) for b in token_bytes)
        except:
            token_str = token_bytes.decode('utf-8', errors='replace')
        new_vocab[token_str] = i
    
    # NOTE: Do NOT add BOS/EOS to new_vocab here!
    # Special tokens are handled separately via token_mapping in _update_tokenizer_json
    # Adding them here with wrong IDs causes vocab size mismatch
    
    # First, copy tokenizer config files from original (to get the structure)
    try:
        old_tokenizer.save_pretrained(output_path)
        print(f"Tokenizer config files saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save full tokenizer config: {e}")
    
    # Save vocab.json (for reference)
    vocab_path = os.path.join(output_path, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(new_vocab, f, ensure_ascii=False, indent=2)
    print(f"New HuggingFace vocab.json (size: {len(new_vocab)}) saved to {vocab_path}")
    
    # Update tokenizer.json with the new vocabulary (this is the key fix!)
    # Returns the actual vocab size and special token IDs for consistency
    actual_vocab_size, special_token_ids = _update_tokenizer_json(
        output_path, new_vocab, token_mapping, old_tokenizer,
        extra_special_tokens=extra_special_tokens
    )
    
    # Update tokenizer config to only have BOS and EOS, remove other special tokens
    if only_bos_eos:
        _update_tokenizer_config_minimal(output_path, old_tokenizer, new_vocab,
                                         actual_vocab_size, special_token_ids,
                                         extra_special_tokens=extra_special_tokens)


def _update_tokenizer_json(output_path, new_vocab, token_mapping, old_tokenizer,
                           extra_special_tokens=None):
    """
    Update tokenizer.json with the new pruned vocabulary.
    This is essential for modern HuggingFace tokenizers (Qwen2, etc.)
    
    Args:
        extra_special_tokens: List of new special token strings to add.
    
    Returns:
        tuple: (actual_vocab_size, special_token_ids dict)
    """
    if extra_special_tokens is None:
        extra_special_tokens = []

    tokenizer_json_path = os.path.join(output_path, 'tokenizer.json')
    if not os.path.exists(tokenizer_json_path):
        print(f"Warning: tokenizer.json not found, skipping update")
        return len(token_mapping), {}
    
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # Get old vocabulary from tokenizer.json
    old_vocab = tokenizer_data.get('model', {}).get('vocab', {})
    if not old_vocab:
        print(f"Warning: No vocab found in tokenizer.json")
        return len(token_mapping), {}
    
    # Build reverse mapping: old_id -> old_token_str
    old_id_to_token = {v: k for k, v in old_vocab.items()}
    
    # Get essential special token IDs and their content
    essential_special_tokens = {}  # old_id -> token_content
    if old_tokenizer is not None:
        if hasattr(old_tokenizer, 'bos_token_id') and old_tokenizer.bos_token_id is not None:
            essential_special_tokens[old_tokenizer.bos_token_id] = old_tokenizer.bos_token
        if hasattr(old_tokenizer, 'eos_token_id') and old_tokenizer.eos_token_id is not None:
            essential_special_tokens[old_tokenizer.eos_token_id] = old_tokenizer.eos_token
        if hasattr(old_tokenizer, 'pad_token_id') and old_tokenizer.pad_token_id is not None:
            essential_special_tokens[old_tokenizer.pad_token_id] = old_tokenizer.pad_token
    
    # Build new vocabulary for tokenizer.json using token_mapping
    # token_mapping[new_id] = old_id
    pruned_vocab = {}
    special_token_new_ids = {}  # token_content -> new_id
    
    for new_id, old_id in enumerate(token_mapping):
        if old_id in old_id_to_token:
            # Regular BPE token
            old_token_str = old_id_to_token[old_id]
            pruned_vocab[old_token_str] = new_id
        elif old_id in essential_special_tokens:
            # Special token (BOS, EOS, PAD)
            token_content = essential_special_tokens[old_id]
            special_token_new_ids[token_content] = new_id
    
    # Update the vocabulary in tokenizer.json
    tokenizer_data['model']['vocab'] = pruned_vocab
    
    # Update merges to only include valid tokens
    if 'merges' in tokenizer_data.get('model', {}):
        old_merges = tokenizer_data['model']['merges']
        valid_tokens = set(pruned_vocab.keys())
        new_merges = []
        for merge in old_merges:
            # Merge format can be either "token1 token2" (string) or ["token1", "token2"] (list)
            if isinstance(merge, str):
                parts = merge.split(' ')
            elif isinstance(merge, list):
                parts = merge
            else:
                continue
            
            if len(parts) == 2:
                # Check if both tokens and the merged result exist in vocabulary
                merged = parts[0] + parts[1]
                if parts[0] in valid_tokens and parts[1] in valid_tokens and merged in valid_tokens:
                    new_merges.append(merge)
        tokenizer_data['model']['merges'] = new_merges
        print(f"Updated merges: {len(old_merges)} -> {len(new_merges)}")
    
    # Update added_tokens - keep BOS/EOS/PAD and add extra special tokens
    new_added_tokens = []
    if 'added_tokens' in tokenizer_data:
        essential_contents = set(essential_special_tokens.values())
        
        for token_info in tokenizer_data['added_tokens']:
            token_content = token_info.get('content', '')
            
            if token_content in essential_contents:
                if token_content in special_token_new_ids:
                    token_info['id'] = special_token_new_ids[token_content]
                    new_added_tokens.append(token_info)
                elif token_content in pruned_vocab:
                    token_info['id'] = pruned_vocab[token_content]
                    new_added_tokens.append(token_info)

    # Append extra special tokens with IDs starting after token_mapping
    extra_base_id = len(token_mapping)
    for i, token_str in enumerate(extra_special_tokens):
        new_id = extra_base_id + i
        special_token_new_ids[token_str] = new_id
        new_added_tokens.append({
            "id": new_id,
            "content": token_str,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })

    tokenizer_data['added_tokens'] = new_added_tokens
    if extra_special_tokens:
        print(f"Updated added_tokens: {len(new_added_tokens)} total "
              f"(including {len(extra_special_tokens)} extra special tokens)")
    else:
        print(f"Updated added_tokens: kept {len(new_added_tokens)} essential special tokens")
    
    # Save updated tokenizer.json
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    total_vocab_size = len(pruned_vocab) + len(special_token_new_ids)
    print(f"Updated tokenizer.json: vocabulary size {len(old_vocab)} -> {len(pruned_vocab)} BPE "
          f"+ {len(special_token_new_ids)} special = {total_vocab_size} total")
    
    return total_vocab_size, special_token_new_ids


def _update_merges_txt(output_path, valid_tokens):
    """Update merges.txt to only include valid tokens."""
    merges_path = os.path.join(output_path, 'merges.txt')
    if not os.path.exists(merges_path):
        return
    
    with open(merges_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # First line is usually a header like "#version: 0.2"
    new_lines = []
    if lines and lines[0].startswith('#'):
        new_lines.append(lines[0])
        lines = lines[1:]
    
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        parts = line.split(' ')
        if len(parts) == 2:
            merged = parts[0] + parts[1]
            if parts[0] in valid_tokens and parts[1] in valid_tokens and merged in valid_tokens:
                new_lines.append(line + '\n')
    
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"Updated merges.txt: {len(lines)} -> {len(new_lines) - 1} merges")


def _update_tokenizer_config_minimal(output_path, old_tokenizer, new_vocab,
                                     actual_vocab_size=None, special_token_ids=None,
                                     extra_special_tokens=None):
    """Update tokenizer config to only use BOS/EOS tokens plus any extra special tokens.
    
    Args:
        special_token_ids: Dict mapping token content -> new token ID (from _update_tokenizer_json)
        extra_special_tokens: List of new special token strings that were added.
    """
    if special_token_ids is None:
        special_token_ids = {}
    if extra_special_tokens is None:
        extra_special_tokens = []
    
    # Also update merges.txt
    _update_merges_txt(output_path, set(new_vocab.keys()))
    
    # Update tokenizer_config.json
    tokenizer_config_path = os.path.join(output_path, 'tokenizer_config.json')
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Keep only BOS and EOS related settings
        bos_token = old_tokenizer.bos_token if hasattr(old_tokenizer, 'bos_token') else None
        eos_token = old_tokenizer.eos_token if hasattr(old_tokenizer, 'eos_token') else None
        
        # Remove other special token settings AND added_tokens_decoder (contains old special tokens)
        keys_to_remove = ['pad_token', 'unk_token', 'sep_token', 'cls_token', 'mask_token',
                          'additional_special_tokens', 'added_tokens_decoder', 'extra_special_tokens']
        for key in keys_to_remove:
            if key in config:
                del config[key]
        
        # Set vocab_size explicitly if provided
        if actual_vocab_size is not None:
            config['vocab_size'] = actual_vocab_size
            print(f"Set vocab_size in tokenizer_config.json: {actual_vocab_size}")
        
        # Set BOS/EOS settings
        config['add_bos_token'] = False
        config['add_eos_token'] = False
        if bos_token:
            config['bos_token'] = bos_token
        if eos_token:
            config['eos_token'] = eos_token
            config['pad_token'] = eos_token

        if extra_special_tokens:
            config['additional_special_tokens'] = list(extra_special_tokens)

        # Build added_tokens_decoder for all special tokens (HF expects this)
        added_tokens_decoder = {}
        for tok_content, tok_id in special_token_ids.items():
            added_tokens_decoder[str(tok_id)] = {
                "content": tok_content,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
        if added_tokens_decoder:
            config['added_tokens_decoder'] = added_tokens_decoder
        
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Updated tokenizer_config.json")
    
    # Update special_tokens_map.json
    special_tokens_map_path = os.path.join(output_path, 'special_tokens_map.json')
    if os.path.exists(special_tokens_map_path):
        bos_token = old_tokenizer.bos_token if hasattr(old_tokenizer, 'bos_token') else None
        eos_token = old_tokenizer.eos_token if hasattr(old_tokenizer, 'eos_token') else None
        
        minimal_map = {}
        if bos_token:
            minimal_map['bos_token'] = bos_token
        if eos_token:
            minimal_map['eos_token'] = eos_token
            minimal_map['pad_token'] = eos_token
        if extra_special_tokens:
            minimal_map['additional_special_tokens'] = list(extra_special_tokens)
        
        with open(special_tokens_map_path, 'w', encoding='utf-8') as f:
            json.dump(minimal_map, f, ensure_ascii=False, indent=2)
        print(f"Updated special_tokens_map.json")
    
    # Update added_tokens.json with all special tokens (BOS/EOS + extras)
    added_tokens_path = os.path.join(output_path, 'added_tokens.json')
    if os.path.exists(added_tokens_path):
        added_map = {}
        for tok_content, tok_id in special_token_ids.items():
            added_map[tok_content] = tok_id
        
        with open(added_tokens_path, 'w', encoding='utf-8') as f:
            json.dump(added_map, f, ensure_ascii=False, indent=2)
        print(f"Updated added_tokens.json: {len(added_map)} tokens")
