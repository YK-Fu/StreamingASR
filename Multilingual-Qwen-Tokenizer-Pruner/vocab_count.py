"""
Parallel-only vocabulary counting with per-file rare token filtering.

This module provides efficient parallel token counting for large datasets,
with optional filtering of rare tokens on a per-file basis.
"""

import os
import warnings
from tqdm import tqdm
import json
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Suppress tokenizer length warnings (we're just counting, not running through model)
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than")

# Global tokenizer for workers (avoid serialization)
_worker_tokenizer = None
_worker_vocab_size = None
_worker_tokenizer_path = None


def _count_batch(args) -> np.ndarray:
    """Count tokens in a batch of lines. Loads tokenizer lazily."""
    global _worker_tokenizer, _worker_vocab_size, _worker_tokenizer_path
    
    lines, tokenizer_path, vocab_size, max_length = args
    
    # Lazy initialization - load tokenizer on first use in this process
    if _worker_tokenizer is None or _worker_tokenizer_path != tokenizer_path:
        from transformers import AutoTokenizer, logging
        logging.set_verbosity_error()
        _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        _worker_tokenizer_path = tokenizer_path
        _worker_vocab_size = vocab_size
    
    counts = np.zeros(vocab_size, dtype=np.int64)
    
    # Parse JSON and extract texts
    texts = []
    for line in lines:
        try:
            data = json.loads(line)
            if 'text' in data and data['text']:
                texts.append(data['text'])
            elif 'sentence' in data and data['sentence']:
                texts.append(data['sentence'])
        except:
            continue
    
    if not texts:
        return counts
    
    # Batch tokenization with truncation to limit memory usage
    encoded = _worker_tokenizer(
        texts, 
        add_special_tokens=False, 
        return_attention_mask=False,
        truncation=True,
        max_length=max_length
    )
    
    # Count with numpy bincount (very fast)
    for token_ids in encoded['input_ids']:
        if token_ids:
            counts += np.bincount(token_ids, minlength=vocab_size)
    
    return counts


def filter_rare_tokens(counts: np.ndarray, percentile: float = 5.0) -> np.ndarray:
    """
    Zero out the rarest X% of tokens within a file's counts.
    
    Args:
        counts: Token counts array for a single file
        percentile: Bottom percentile to zero out (default 5%)
    
    Returns:
        Filtered counts with rare tokens zeroed
    """
    # Find indices of tokens that were actually used (count > 0)
    used_mask = counts > 0
    used_indices = np.where(used_mask)[0]
    
    if len(used_indices) == 0:
        return counts
    
    # Calculate how many tokens to zero (bottom X%)
    num_to_zero = max(1, int(len(used_indices) * percentile / 100))
    
    # Get counts of used tokens and find the threshold
    used_counts = counts[used_indices]
    sorted_indices = np.argsort(used_counts)
    
    # Zero out the bottom X% (rarest tokens in this file)
    indices_to_zero = used_indices[sorted_indices[:num_to_zero]]
    filtered = counts.copy()
    filtered[indices_to_zero] = 0
    
    return filtered


def get_counts_parallel(file_name: str, vocab_size: int, tokenizer_path: str, 
                        batch_size: int = 5000, num_workers: int = 16,
                        max_length: int = 8192, executor=None,
                        filter_rare_percentile: float = None) -> np.ndarray:
    """Count tokens in a file using parallel batch processing.
    
    Args:
        file_name: Path to JSONL file
        vocab_size: Size of vocabulary
        tokenizer_path: Path to tokenizer
        batch_size: Number of lines per batch (default 5000)
        num_workers: Number of parallel workers (default 16)
        max_length: Maximum tokens per text (default 8192, reduces memory usage)
        executor: Optional existing ProcessPoolExecutor to reuse
        filter_rare_percentile: If set, zero out bottom X% of tokens for this file
    
    Returns:
        Token counts array (optionally filtered)
    """
    if num_workers is None:
        num_workers = min(8, mp.cpu_count())
    
    # Stream file and process in batches (don't load entire file into memory)
    def batch_generator():
        batch = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                batch.append(line)
                if len(batch) >= batch_size:
                    yield (batch, tokenizer_path, vocab_size, max_length)
                    batch = []
            if batch:
                yield (batch, tokenizer_path, vocab_size, max_length)
    
    batches = list(batch_generator())
    
    # Process batches in parallel
    total_counts = np.zeros(vocab_size, dtype=np.int64)
    
    if executor is not None:
        for batch_counts in tqdm(executor.map(_count_batch, batches), 
                                  total=len(batches), 
                                  desc=f"Processing {os.path.basename(file_name)}"):
            total_counts += batch_counts
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as exec:
            for batch_counts in tqdm(exec.map(_count_batch, batches), 
                                      total=len(batches), 
                                      desc=f"Processing {os.path.basename(file_name)}"):
                total_counts += batch_counts
    
    # Apply per-file rare token filtering if requested
    if filter_rare_percentile is not None and filter_rare_percentile > 0:
        num_unique_before = np.sum(total_counts > 0)
        total_counts = filter_rare_tokens(total_counts, filter_rare_percentile)
        num_unique_after = np.sum(total_counts > 0)
        print(f"  Filtered: {num_unique_before:,} -> {num_unique_after:,} unique tokens "
              f"(removed {num_unique_before - num_unique_after:,} rare tokens)")
    
    return total_counts


def count_freq(data_path: str, vocab_size: int, tokenizer_path: str, 
               output_path: str, inherit_vocab_count: str = None,
               batch_size: int = 5000, num_workers: int = 16, 
               max_length: int = 8192,
               filter_rare_percentile: float = None) -> list:
    """
    Count token frequencies across all files in data_path using parallel processing.
    
    Args:
        data_path: Directory containing JSONL files
        vocab_size: Size of vocabulary
        tokenizer_path: Path to tokenizer (required)
        output_path: Directory to save vocab_counts.torch
        inherit_vocab_count: Optional path to existing vocab counts to add
        batch_size: Number of lines per batch (default 5000)
        num_workers: Number of parallel workers (default 16)
        max_length: Maximum tokens per text (default 8192)
        filter_rare_percentile: If set, zero out bottom X% of tokens per file
    
    Returns:
        List of token counts
    """
    if tokenizer_path is None:
        raise ValueError("tokenizer_path is required for parallel processing")
    
    vocab_counts = np.zeros(vocab_size, dtype=np.int64)
    
    files = [f for f in os.listdir(data_path) if f.endswith('.jsonl')]
    
    print(f"Using {num_workers} workers, max_length={max_length} tokens per text")
    if filter_rare_percentile:
        print(f"Per-file filtering: removing bottom {filter_rare_percentile}% rare tokens")
    
    # Reuse executor across files to save memory
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        for path in tqdm(files, desc="Processing files"):
            file_counts = get_counts_parallel(
                os.path.join(data_path, path),
                vocab_size,
                tokenizer_path,
                batch_size=batch_size,
                num_workers=num_workers,
                max_length=max_length,
                executor=executor,
                filter_rare_percentile=filter_rare_percentile
            )
            vocab_counts += file_counts

    # Add inherited vocab counts if provided
    if inherit_vocab_count is not None:
        if os.path.exists(inherit_vocab_count):
            print(f"==> Load inherit_vocab_count and add it to current vocab_counts: path({inherit_vocab_count})")
            loaded_counts = torch.load(inherit_vocab_count)
            assert len(loaded_counts) == vocab_size, \
                f"inherit_vocab_count (size: {len(loaded_counts)}) should have the same vocab size {vocab_size}"
            vocab_counts += np.array(loaded_counts, dtype=np.int64)
        else:
            print(f"==> No valid inherit vocabulary count path, skip inheritance!")
    
    # Save vocab_counts (convert to list for compatibility)
    torch.save(vocab_counts.tolist(), os.path.join(output_path, 'vocab_counts.torch'))
    return vocab_counts.tolist()


def _process_recursive_chunk(args):
    """Process a chunk of tokens for recursive counting. Module-level for pickling."""
    chunk, bytes_to_idx, vocab_size = args
    local_counts = np.zeros(vocab_size, dtype=np.int64)
    for i, token_bytes, t_count in chunk:
        b_len = len(token_bytes)
        for j in range(1, b_len):
            for k in range(b_len + 1 - j):
                sub_token = token_bytes[k:k + j]
                idx = bytes_to_idx.get(sub_token)
                if idx is not None:
                    local_counts[idx] += t_count
    return local_counts


def count_recursive_parallel(vocab_size: int, vocab_counts, old_bytes_list: list, 
                             num_workers: int = None) -> list:
    """
    Parallel counting of recursive/inherited token frequencies.
    
    For each token, find all substrings that are also valid tokens,
    and add the parent token's count to those sub-tokens.
    
    Args:
        vocab_size: Size of vocabulary
        vocab_counts: Token counts (list or array)
        old_bytes_list: List of byte sequences for each token
        num_workers: Number of parallel workers (default: min(8, cpu_count))
    
    Returns:
        List of recursive counts
    """
    if num_workers is None:
        num_workers = min(8, mp.cpu_count())
    
    # Build a hash map for O(1) lookup
    bytes_to_idx = {token_bytes: idx for idx, token_bytes in enumerate(old_bytes_list)}
    
    # Convert vocab_counts to numpy if needed
    if not isinstance(vocab_counts, np.ndarray):
        vocab_counts = np.array(vocab_counts, dtype=np.int64)
    
    # Filter to only tokens with count > 0 and length > 1
    work_items = [
        (i, old_bytes_list[i], int(vocab_counts[i]))
        for i in range(len(old_bytes_list))
        if vocab_counts[i] > 0 and len(old_bytes_list[i]) > 1
    ]
    
    print(f"Processing {len(work_items):,} tokens with {num_workers} workers...")
    
    # Split work into chunks
    chunk_size = max(1, len(work_items) // num_workers)
    chunks = [(work_items[i:i + chunk_size], bytes_to_idx, vocab_size) 
              for i in range(0, len(work_items), chunk_size)]
    
    # Process in parallel
    recursive_counts = np.zeros(vocab_size, dtype=np.int64)
    
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        for result in tqdm(executor.map(_process_recursive_chunk, chunks), 
                          total=len(chunks), desc="Counting recursive (parallel)"):
            recursive_counts += result
    
    return recursive_counts.tolist()
