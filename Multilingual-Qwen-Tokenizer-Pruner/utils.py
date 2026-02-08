"""
Utility functions for vocabulary loading and tokenizer format detection.
"""

import os
import base64
import torch


def get_bpe_file(root_path):
    """Find tiktoken BPE file in the model directory. Returns None if not found."""
    all_matchs = []
    for file_name in os.listdir(root_path):
        if file_name.endswith('tiktoken'):
            all_matchs.append(os.path.join(root_path, file_name))
    if len(all_matchs) == 1:
        return all_matchs[0]
    elif len(all_matchs) > 1:
        raise ValueError(f"Multiple tiktoken bpe files found: {all_matchs}")
    return None  # No tiktoken file found


def detect_tokenizer_format(model_path):
    """
    Detect whether the tokenizer is tiktoken or HuggingFace format.
    Returns: 'tiktoken' or 'huggingface'
    """
    tiktoken_file = get_bpe_file(model_path)
    if tiktoken_file is not None:
        return 'tiktoken', tiktoken_file
    
    # Check for HuggingFace tokenizer files
    hf_files = ['tokenizer.json', 'vocab.json', 'tokenizer_config.json']
    for f in hf_files:
        if os.path.exists(os.path.join(model_path, f)):
            return 'huggingface', None
    
    raise ValueError(f"Could not detect tokenizer format in {model_path}")


def load_bytes_list_from_tiktoken(tiktoken_file):
    """Load vocabulary as bytes list from tiktoken BPE file."""
    with open(tiktoken_file, "rb") as f:
        contents = f.read()
    bytes_list = [
        base64.b64decode(token) 
        for token, rank in (line.split() for line in contents.splitlines() if line)
    ]
    return bytes_list


def load_bytes_list_from_huggingface(tokenizer, vocab_size):
    """
    Load vocabulary as bytes list from HuggingFace tokenizer.
    Converts token strings to their byte representations.
    """
    # Get vocabulary: token_str -> token_id
    vocab = tokenizer.get_vocab()
    
    # Create reverse mapping: token_id -> token_str
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Build bytes list in order of token IDs
    bytes_list = []
    
    # For byte-level BPE tokenizers (like GPT-2, Qwen), we need to handle the byte encoding
    # Most HuggingFace tokenizers use a byte-to-unicode mapping
    byte_decoder = _get_byte_decoder()
    
    for i in range(vocab_size):
        if i in id_to_token:
            token_str = id_to_token[i]
            
            # Try to convert token string to bytes
            try:
                # For byte-level BPE: decode using byte_decoder
                token_bytes = bytes([byte_decoder.get(c, ord(c)) for c in token_str])
            except (KeyError, ValueError):
                # Fallback: encode as UTF-8
                try:
                    token_bytes = token_str.encode('utf-8')
                except:
                    token_bytes = b''
            
            bytes_list.append(token_bytes)
        else:
            # Special tokens or missing tokens
            bytes_list.append(b'')
    
    # Remove trailing empty entries (special tokens are handled separately)
    while bytes_list and bytes_list[-1] == b'':
        bytes_list.pop()
    
    return bytes_list


def _get_byte_decoder():
    """
    Get byte decoder mapping (unicode char -> byte value).
    This is the inverse of GPT-2's bytes_to_unicode() function.
    Used by many byte-level BPE tokenizers including Qwen.
    """
    # This is the standard GPT-2 byte encoding
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
    return dict(zip(cs, bs))


def load_vocabulary_bytes(model_path, tokenizer=None, vocab_size=None):
    """
    Load vocabulary as bytes list from either tiktoken or HuggingFace format.
    
    Args:
        model_path: Path to model directory
        tokenizer: HuggingFace tokenizer (required for HuggingFace format)
        vocab_size: Vocabulary size (required for HuggingFace format)
    
    Returns:
        bytes_list: List of byte sequences for each token
        format_type: 'tiktoken' or 'huggingface'
    """
    format_type, tiktoken_file = detect_tokenizer_format(model_path)
    
    if format_type == 'tiktoken':
        print(f"==> Detected tiktoken format: {tiktoken_file}")
        bytes_list = load_bytes_list_from_tiktoken(tiktoken_file)
    else:
        print(f"==> Detected HuggingFace tokenizer format")
        if tokenizer is None or vocab_size is None:
            raise ValueError("tokenizer and vocab_size required for HuggingFace format")
        bytes_list = load_bytes_list_from_huggingface(tokenizer, vocab_size)
    
    print(f"==> Loaded {len(bytes_list)} tokens from vocabulary")
    return bytes_list, format_type
