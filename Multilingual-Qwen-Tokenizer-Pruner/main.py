"""
Multilingual Qwen Tokenizer Pruner

Main entry point for vocabulary pruning with parallel processing
and optional per-file rare token filtering.
"""

import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from vocab_count import count_freq, count_recursive_parallel, scan_cjk_chars
from vocab_save import get_new_vocab_and_map, save_vocab, reduce_to_target_size, filter_long_tokens, filter_multichar_cjk_tokens, _get_byte_encoder
from model_save import saving_updated_qwen, saving_updated_qwenvl
from utils import load_vocabulary_bytes


def load_tokenizer_internals(old_model_path):
    """Return (str2id, id2str, merge_parent) from the original tokenizer.json.

    merge_parent maps each merged token string -> (a_str, b_str), the unique pair
    whose concatenation produces it (byte-level BPE creates one token per merge).
    """
    with open(os.path.join(old_model_path, 'tokenizer.json'), 'r', encoding='utf-8') as f:
        tj = json.load(f)
    str2id = tj['model']['vocab']                 # byte-level str -> old id
    id2str = {v: k for k, v in str2id.items()}
    merge_parent = {}
    for mg in tj['model']['merges']:
        a, b = (mg.split(' ') if isinstance(mg, str) else (mg[0], mg[1]))
        merge_parent[a + b] = (a, b)
    return str2id, id2str, merge_parent


def force_keep_closure(token_str, merge_parent, str2id, vocab_counts, recur_counts):
    """Force-keep token_str and every token on its byte-level merge path (down to
    bytes), so the BPE merge path survives pruning and the token stays reachable."""
    stack = [token_str]
    while stack:
        t = stack.pop()
        tid = str2id.get(t)
        if tid is not None and vocab_counts[tid] + recur_counts[tid] == 0:
            vocab_counts[tid] = 1
        if t in merge_parent:
            stack.extend(merge_parent[t])


def select_per_language_keep(per_lang_counts, alive_mask, max_size):
    """Keep at most `max_size` tokens per language (its top-N by its own counts,
    restricted to tokens that survived earlier filters via alive_mask), unioned
    across languages.

    Returns (keep_ids set, info dict of stem -> kept count).
    """
    keep, info = set(), {}
    for stem, counts in per_lang_counts.items():
        c = counts.astype(np.float64) * alive_mask
        order = np.argsort(c)[::-1]
        n_nz = int((c[order] > 0).sum())
        K = min(max_size, n_nz)
        keep.update(int(i) for i in order[:K])
        info[stem] = K
    return keep, info


def build_vocab_padding_tokens(vocab_size, multiple, existing_tokens=()):
    """Return reserved special tokens that pad ``vocab_size`` to ``multiple``.

    Padding tokens are appended at the very end of the tokenizer so all real
    token IDs remain unchanged. Names are chosen deterministically while
    avoiding collisions with source and user-provided special tokens.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if multiple <= 0:
        raise ValueError(f"pad_vocab_multiple must be positive, got {multiple}")

    padding_count = (-vocab_size) % multiple
    if padding_count == 0:
        return []

    used = set(existing_tokens)
    padding_tokens = []
    candidate_idx = 0
    while len(padding_tokens) < padding_count:
        token = f"<|vocab_padding_{candidate_idx:03d}|>"
        candidate_idx += 1
        if token in used:
            continue
        used.add(token)
        padding_tokens.append(token)
    return padding_tokens


def prepare_native_char_tokens(chars, old_tokenizer, internals, vocab_counts, recur_counts):
    """Guarantee every char in `chars` is a single *native BPE* token in the pruned model.

    Two cases per character (no added tokens are used):
      - Already one token in original Qwen: force-keep its full merge-decomposition
        closure (the token + every intermediate token down to bytes) so the byte-level
        merge path survives pruning and BPE rebuilds it. Original id/embedding reused.
      - Splits into k>1 tokens: fold the fragments left-to-right into the char token,
        creating k-1 new vocab entries (k-2 intermediates + the char) and k-1 new
        merges appended at lowest priority. Each new entry's embedding is the mean of
        the original-Qwen fragment rows it spans.

    vocab_counts/recur_counts are bumped in place to force-keep needed tokens.

    Returns (native_tokens, native_merges):
      native_tokens: ordered list of NEW BPE vocab entries to append:
          {"str": <byte-level token string>,
           "init_ids": [old_ids whose embedding rows to average]}
      native_merges: ordered, de-duplicated list of [a_str, b_str] merge rules to
          append at lowest priority so BPE rebuilds each native token.
    """
    str2id, id2str, merge_parent = internals

    def keep_closure(token_str):
        force_keep_closure(token_str, merge_parent, str2id, vocab_counts, recur_counts)

    # Force-keep all 256 base byte tokens so the tokenizer stays byte-level OOV-free
    # (the pruner otherwise drops unused byte tokens, breaking rare bytes/scripts).
    n_bytes = 0
    for byte_char in _get_byte_encoder().values():
        bid = str2id.get(byte_char)
        if bid is not None and vocab_counts[bid] + recur_counts[bid] == 0:
            vocab_counts[bid] = 1
            n_bytes += 1
    print(f"==> Force-kept {n_bytes} previously-unused base byte tokens (OOV-free)")

    print(f"==> Guaranteeing native single-token representation for {len(chars):,} characters")
    native_tokens, native_merges = [], []
    seen_new, seen_merge = set(), set()
    n_keep = n_two = n_three = 0
    for ch in chars:
        ids = old_tokenizer.encode(ch, add_special_tokens=False)
        if not ids:
            continue
        strs = [id2str[i] for i in ids]
        for s in strs:                           # make every fragment reachable
            keep_closure(s)
        if len(ids) == 1:
            n_keep += 1
            continue
        # Left-fold fragments into the char token, adding a merge (and a new vocab
        # entry for each not-yet-existing intermediate / the char itself).
        cur, cur_ids = strs[0], [ids[0]]
        for j in range(1, len(strs)):
            nxt, nxt_ids = cur + strs[j], cur_ids + [ids[j]]
            if (cur, strs[j]) not in seen_merge:
                native_merges.append([cur, strs[j]])
                seen_merge.add((cur, strs[j]))
            if nxt in str2id:
                keep_closure(nxt)                # existing token: retain it
            elif nxt not in seen_new:
                native_tokens.append({'str': nxt, 'init_ids': list(nxt_ids)})
                seen_new.add(nxt)
            cur, cur_ids = nxt, nxt_ids
        n_two += (len(ids) == 2)
        n_three += (len(ids) >= 3)
    print(f"   already native single token : {n_keep:,}")
    print(f"   2-token chars merged (native): {n_two:,}")
    print(f"   3+-token chars merged (native): {n_three:,}")
    print(f"   new native vocab entries: {len(native_tokens):,}  new merges: {len(native_merges):,}")
    return native_tokens, native_merges


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
    parser.add_argument('--inherit_per_lang_counts', type=str, default=None,
                        help='Path to an existing per_lang_counts.npz for --per_lang_target_size. '
                             'Allows per-language pruning without --support_data.')
    parser.add_argument('--target_size', type=int, default=None,
                        help='Global target vocabulary size (optional)')
    parser.add_argument('--per_lang_target_size', type=int, default=None,
                        help='Maximum vocab size PER LANGUAGE (instead of a global --target_size). '
                             'Each language keeps at most this many of its most frequent tokens; '
                             'the kept sets are unioned. Requires --support_data or '
                             '--inherit_per_lang_counts. CJK char / byte / '
                             'special tokens are kept on top of this budget.')
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
    parser.add_argument('--filter_multichar_cjk', action='store_true',
                        help='CJK char-level mode: (1) remove CJK tokens longer than 1 character, '
                             'and (2) guarantee every CJK-family character in --support_data is a '
                             'single native BPE token (merging sub-tokens with averaged embeddings).')
    parser.add_argument('--add_special_tokens', type=str, default=None,
                        help='Path to text file with new special tokens to add (one per line)')
    parser.add_argument('--pad_vocab_multiple', type=int, default=None,
                        help='Append reserved special tokens until the final tokenizer/model '
                             'vocabulary is divisible by this value (for example: 64).')
    args = parser.parse_args()
    if args.pad_vocab_multiple is not None and args.pad_vocab_multiple <= 0:
        parser.error('--pad_vocab_multiple must be positive')
    
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
    
    # Filter multi-character CJK tokens if requested
    if args.filter_multichar_cjk:
        print(f"==> Filtering multi-character CJK tokens")
        vocab_counts, recur_counts, _ = filter_multichar_cjk_tokens(
            vocab_counts=vocab_counts,
            recur_counts=recur_counts,
            old_bytes_list=old_bytes_list
        )

    # Per-language budgeting (--per_lang_target_size): keep top-K tokens per language
    # (by that language's own counts) + their merge closures, instead of a global cut.
    internals = load_tokenizer_internals(args.old_model_path)
    if args.per_lang_target_size is not None:
        if args.target_size is not None:
            raise ValueError("Use either --target_size or --per_lang_target_size, not both")
        if args.support_data is not None:
            per_lang_counts_path = os.path.join(args.new_model_path, 'per_lang_counts.npz')
        elif args.inherit_per_lang_counts is not None:
            per_lang_counts_path = args.inherit_per_lang_counts
        else:
            raise ValueError(
                "--per_lang_target_size requires --support_data or "
                "--inherit_per_lang_counts"
            )
        if not os.path.isfile(per_lang_counts_path):
            raise ValueError(f"Per-language counts file not found: {per_lang_counts_path}")
        print(f"==> Max vocab size per language: {args.per_lang_target_size:,}")
        print(f"==> Loading per-language counts from: {per_lang_counts_path}")
        with np.load(per_lang_counts_path) as loaded_per_lang_counts:
            per_lang_counts = {
                stem: counts for stem, counts in loaded_per_lang_counts.items()
            }
        invalid_counts = [
            stem for stem, counts in per_lang_counts.items()
            if counts.ndim != 1 or len(counts) != old_vocab_size
        ]
        if invalid_counts:
            raise ValueError(
                "Each per-language count array must have one entry per original token "
                f"({old_vocab_size}); invalid entries: {', '.join(sorted(invalid_counts))}"
            )
        if not per_lang_counts:
            raise ValueError(f"No per-language counts found in: {per_lang_counts_path}")
        alive_mask = (np.array(vocab_counts, dtype=np.int64) > 0).astype(np.float64)
        keep_ids, info = select_per_language_keep(per_lang_counts, alive_mask,
                                                  args.per_lang_target_size)
        for stem in sorted(info):
            print(f"   {stem:<8} keep top {info[stem]:,}")
        # Rewrite counts to keep only the selected ids + their merge closures.
        str2id, id2str, merge_parent = internals
        vocab_counts = [0] * old_vocab_size
        recur_counts = [0] * old_vocab_size
        for tid in keep_ids:
            vocab_counts[tid] = 1
        for tid in keep_ids:
            s = id2str.get(tid)
            if s is not None:
                force_keep_closure(s, merge_parent, str2id, vocab_counts, recur_counts)
        print(f"   selected {len(keep_ids):,} tokens, {sum(1 for c in vocab_counts if c>0):,} after merge-closure")
    elif args.target_size is not None:
        # Reduce vocab to a global target size
        print(f"==> Reducing vocab to target size: {args.target_size:,}")
        vocab_counts, recur_counts = reduce_to_target_size(
            old_vocab_size=old_vocab_size,
            target_vocab_size=args.target_size,
            vocab_counts=vocab_counts,
            recur_counts=recur_counts,
            old_bytes_list=old_bytes_list
        )

    # CJK char-level mode (driven by --filter_multichar_cjk): guarantee every
    # CJK-family character in the support data is a single native BPE token.
    native_tokens, native_merges = [], []
    if args.filter_multichar_cjk and args.support_data is not None:
        cjk_chars = scan_cjk_chars(args.support_data, num_workers=args.num_workers)
        native_tokens, native_merges = prepare_native_char_tokens(
            cjk_chars, old_tokenizer, internals, vocab_counts, recur_counts)
    native_token_strs = [t['str'] for t in native_tokens]
    merge_init = [t['init_ids'] for t in native_tokens]

    # Read extra special tokens file if provided
    extra_special_tokens = []
    if args.add_special_tokens is not None:
        with open(args.add_special_tokens, 'r', encoding='utf-8') as f:
            extra_special_tokens = [line.strip() for line in f if line.strip()]
        print(f"==> Will add {len(extra_special_tokens)} extra special tokens from {args.add_special_tokens}")
    num_user_extra_special = len(extra_special_tokens)

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
    # HF-compatible ID layout:
    # [mapped BPE][native merged-char BPE][retained special][extra special]
    # [vocab-padding special]. Padding is calculated only after every real token
    # has been counted, so it cannot shift an existing token ID.
    num_mapped_bpe = len(new_bytes_list)
    unpadded_vocab_size = (
        len(mapping_new2old) + len(extra_special_tokens) + len(native_token_strs)
    )
    padding_special_tokens = []
    if args.pad_vocab_multiple is not None:
        padding_special_tokens = build_vocab_padding_tokens(
            unpadded_vocab_size,
            args.pad_vocab_multiple,
            existing_tokens=set(old_tokenizer.get_vocab()) | set(extra_special_tokens),
        )
        extra_special_tokens.extend(padding_special_tokens)
        print(
            f"==> Vocabulary alignment: {unpadded_vocab_size:,} + "
            f"{len(padding_special_tokens)} reserved token(s) = "
            f"{unpadded_vocab_size + len(padding_special_tokens):,} "
            f"(multiple of {args.pad_vocab_multiple})"
        )
    new_vocab_size = unpadded_vocab_size + len(padding_special_tokens)

    # Save vocabulary files
    save_vocab(
        new_bytes_list,
        mapping_new2old,
        args.new_model_path,
        tokenizer_format=tokenizer_format,
        old_tokenizer=old_tokenizer,
        extra_special_tokens=extra_special_tokens,
        native_tokens=native_token_strs,
        native_merges=native_merges
    )

    # Update and save model checkpoint
    print(f"==> Updating model checkpoint")
    if 'visual' in old_model.config.__dict__:
        print(f"  Detected Qwen-VL model")
        saving_updated_qwenvl(old_model, new_vocab_size, mapping_new2old, args.new_model_path,
                              num_extra_special=len(extra_special_tokens), merge_init=merge_init,
                              num_mapped_bpe=num_mapped_bpe,
                              num_padding_special=len(padding_special_tokens))
    else:
        print(f"  Detected standard Qwen model")
        saving_updated_qwen(old_model, new_vocab_size, mapping_new2old, args.new_model_path,
                            num_extra_special=len(extra_special_tokens), merge_init=merge_init,
                            num_mapped_bpe=num_mapped_bpe,
                            num_padding_special=len(padding_special_tokens))
    
    print(f"\n{'='*50}")
    print(f"Vocabulary pruning complete!")
    print(f"  Original size: {old_vocab_size:,}")
    print(f"  New size:      {new_vocab_size:,}")
    if num_user_extra_special:
        print(f"    (includes {num_user_extra_special} user-provided special tokens)")
    if padding_special_tokens:
        print(f"    (includes {len(padding_special_tokens)} vocabulary-padding tokens)")
    if native_token_strs:
        print(f"    (includes {len(native_token_strs)} native merged CJK-char tokens)")
    print(f"  Reduction:     {old_vocab_size - new_vocab_size:,} tokens ({100*(old_vocab_size-new_vocab_size)/old_vocab_size:.1f}%)")
    print(f"  Output path:   {args.new_model_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
