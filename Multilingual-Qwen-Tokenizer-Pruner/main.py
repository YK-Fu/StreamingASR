"""
Multilingual Qwen Tokenizer Pruner

Main entry point for vocabulary pruning with parallel processing
and optional per-file rare token filtering.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from vocab_count import count_freq, count_recursive_parallel
from vocab_save import get_new_vocab_and_map, save_vocab, reduce_to_target_size, filter_long_tokens
from model_save import saving_updated_qwen, saving_updated_qwenvl
from utils import load_vocabulary_bytes


def main():
    print('============ Multilingual Qwen Vocabulary Pruning ==========')
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Prune Qwen tokenizer vocabulary')
    parser.add_argument('--old_model_path', type=str, required=True,
                        help='Path to original model/tokenizer')
    parser.add_argument('--new_model_path', type=str, required=True,
                        help='Output path for pruned model')
    parser.add_argument('--support_data', type=str, default=None,
                        help='Path to directory containing JSONL files for counting')
    parser.add_argument('--inherit_vocab_count', type=str, default=None,
                        help='Path to existing vocab_counts.torch to inherit from')
    parser.add_argument('--target_size', type=int, default=None,
                        help='Target vocabulary size (optional)')
    parser.add_argument('--filter_rare_percentile', type=float, default=None,
                        help='Zero out bottom X%% of tokens per file (e.g., 5 for 5%%)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size for processing (default: 5000)')
    parser.add_argument('--max_length', type=int, default=8192,
                        help='Max tokens per text (default: 8192)')
    parser.add_argument('--max_token_length', type=int, default=None,
                        help='Filter out tokens with more than N bytes/characters (e.g., 10)')
    args = parser.parse_args()
    
    # Validate: need at least one source of vocabulary counts
    has_data_source = (args.support_data is not None) or (args.inherit_vocab_count is not None)
    if not has_data_source:
        raise ValueError("Must provide at least one of: --support_data or --inherit_vocab_count")

    # Create output directory
    if not os.path.exists(args.new_model_path):
        os.makedirs(args.new_model_path)
        print(f"==> Created output folder: {args.new_model_path}")
    
    # Load old model and tokenizer
    print(f"==> Loading model and tokenizer from: {args.old_model_path}")
    old_model = AutoModelForCausalLM.from_pretrained(args.old_model_path, trust_remote_code=True)
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
    old_vocab_size = old_model.config.__dict__['vocab_size']
    print(f"Original vocabulary size: {old_vocab_size:,}")
    
    # Count token frequencies
    if args.support_data is not None:
        print(f"==> Counting token frequencies from: {args.support_data}")
        vocab_counts = count_freq(
            data_path=args.support_data, 
            vocab_size=old_vocab_size, 
            tokenizer_path=args.old_model_path,
            output_path=args.new_model_path, 
            inherit_vocab_count=args.inherit_vocab_count,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            filter_rare_percentile=args.filter_rare_percentile
        )
    elif args.inherit_vocab_count is not None and os.path.exists(args.inherit_vocab_count):
        # Load vocab counts directly from inherited file
        print(f"==> Loading vocab counts from: {args.inherit_vocab_count}")
        vocab_counts = torch.load(args.inherit_vocab_count)
        if isinstance(vocab_counts, torch.Tensor):
            vocab_counts = vocab_counts.tolist()
        assert len(vocab_counts) == old_vocab_size, \
            f"inherit_vocab_count size ({len(vocab_counts)}) != vocab_size ({old_vocab_size})"
        print(f"Loaded {sum(1 for c in vocab_counts if c > 0):,} non-zero token counts")
    else:
        vocab_counts = [0] * old_vocab_size
        
    # Load vocabulary bytes (supports both tiktoken and HuggingFace formats)
    old_bytes_list, tokenizer_format = load_vocabulary_bytes(
        model_path=args.old_model_path,
        tokenizer=old_tokenizer,
        vocab_size=old_vocab_size
    )
    
    # Calculate recursive sub-token counts
    print(f"==> Computing recursive sub-token counts")
    recur_counts = count_recursive_parallel(
        vocab_size=old_vocab_size, 
        vocab_counts=vocab_counts, 
        old_bytes_list=old_bytes_list,
        num_workers=args.num_workers
    )
    
    # Filter out long tokens if specified
    if args.max_token_length is not None:
        print(f"==> Filtering tokens longer than {args.max_token_length} bytes")
        vocab_counts, recur_counts, _ = filter_long_tokens(
            vocab_counts=vocab_counts,
            recur_counts=recur_counts,
            old_bytes_list=old_bytes_list,
            max_length=args.max_token_length
        )
    
    # Reduce vocab to target size if specified
    if args.target_size is not None:
        print(f"==> Reducing vocab to target size: {args.target_size:,}")
        vocab_counts, recur_counts = reduce_to_target_size(
            old_vocab_size=old_vocab_size, 
            target_vocab_size=args.target_size, 
            vocab_counts=vocab_counts, 
            recur_counts=recur_counts, 
            old_bytes_list=old_bytes_list
        )
    
    # Get new vocabulary and mapping
    print(f"==> Building new vocabulary")
    new_bytes_list, mapping_new2old = get_new_vocab_and_map(
        old_bytes_list=old_bytes_list, 
        old_vocab_size=old_vocab_size,
        vocab_counts=vocab_counts, 
        recur_counts=recur_counts,
        old_tokenizer=old_tokenizer,
        only_essential_special_tokens=True  # Only BOS, EOS, PAD
    )
    new_vocab_size = len(mapping_new2old)
    
    # Save vocabulary files
    save_vocab(
        new_bytes_list, 
        mapping_new2old, 
        args.new_model_path, 
        tokenizer_format=tokenizer_format, 
        old_tokenizer=old_tokenizer
    )

    # Update and save model checkpoint
    print(f"==> Updating model checkpoint")
    if 'visual' in old_model.config.__dict__:
        print(f"  Detected Qwen-VL model")
        saving_updated_qwenvl(old_model, new_vocab_size, mapping_new2old, args.new_model_path)
    else:
        print(f"  Detected standard Qwen model")
        saving_updated_qwen(old_model, new_vocab_size, mapping_new2old, args.new_model_path)
    
    print(f"\n{'='*50}")
    print(f"Vocabulary pruning complete!")
    print(f"  Original size: {old_vocab_size:,}")
    print(f"  New size:      {new_vocab_size:,}")
    print(f"  Reduction:     {old_vocab_size - new_vocab_size:,} tokens ({100*(old_vocab_size-new_vocab_size)/old_vocab_size:.1f}%)")
    print(f"  Output path:   {args.new_model_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
