#!/usr/bin/env python3
"""
Convert a trained CausalWhisperDistilModel checkpoint to HybridRNNTCTCWhisperLMModel.

This script takes the distillation model (with trained student encoder) and creates
a full RNNT model by:
1. Copying student encoder weights to the RNNT encoder
2. Optionally copying CTC decoder weights
3. Initializing the Qwen decoder from HuggingFace (or leaving random)

Usage examples:
    # Convert distill model to RNNT, initialize decoder from HF Qwen
    python convert_distill_to_rnnt.py \
        --distill-checkpoint distill_model.nemo \
        --qwen Qwen/Qwen2.5-0.5B \
        --config conf/hybrid_transducer_ctc.yaml \
        --output rnnt_model.nemo

"""

import argparse
import os
import tarfile
import tempfile
import torch
from typing import Dict, Optional
from collections import OrderedDict
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from nemo.utils.trainer_utils import resolve_trainer_cfg


def load_nemo_checkpoint(nemo_path: str) -> Dict[str, torch.Tensor]:
    """
    Load state_dict from a .nemo file (tarball format).
    
    Args:
        nemo_path: Path to .nemo file
    
    Returns:
        State dict extracted from the .nemo tarball
    """
    if tarfile.is_tarfile(nemo_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(nemo_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            weights_path = os.path.join(tmpdir, "model_weights.ckpt")
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location="cpu")
            else:
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith(".ckpt"):
                            checkpoint = torch.load(os.path.join(root, f), map_location="cpu")
                            break
                    else:
                        continue
                    break
                else:
                    raise ValueError(f"No .ckpt file found in {nemo_path}")
            
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    return checkpoint["state_dict"]
                return checkpoint
            return checkpoint
    else:
        checkpoint = torch.load(nemo_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint


def convert_qwen_decoder_weights(
    hf_model_path: str,
    vocab_size: int,
    tie_weights: bool = True,
) -> tuple:
    """
    Convert HuggingFace Qwen to LLMDecoder state_dict (no prefix).
    
    Args:
        hf_model_path: Path or HuggingFace model name
        vocab_size: Vocabulary size (HF Qwen may have more tokens than needed)
        tie_weights: Whether to tie word embeddings
    
    Returns:
        Tuple of (decoder_state_dict, llm_head_weight or None)
    """
    from transformers import AutoModelForCausalLM
    
    print(f"Loading Qwen model from {hf_model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, trust_remote_code=True)
    hf_state = hf_model.state_dict()
    
    converted = OrderedDict()
    lm_head_weight = None
    
    for key, value in hf_state.items():
        if key.startswith("model."):
            if 'embed_tokens' in key:
                value = value[:vocab_size]
            new_key = key[6:]
            converted[new_key] = value
    
    if not tie_weights:
        if "lm_head.weight" in hf_state:
            lm_head_weight = hf_state["lm_head.weight"][:vocab_size]
            print("Using lm_head.weight for untied projection head")
        elif "embed_tokens.weight" in converted:
            lm_head_weight = converted["embed_tokens.weight"].clone()
            print("Copying embed_tokens.weight for untied projection head")
    else:
        print("Using tied weights - projection head will share embed_tokens.weight")
    
    print(f"Qwen decoder conversion complete. Total keys: {len(converted)}")
    return converted, lm_head_weight


def load_distill_weights_to_rnnt(distill_state, model, copy_ctc_decoder=True):
    """
    Load distillation model student weights into RNNT model encoder.
    
    Args:
        distill_state: State dict from distillation model
        model: HybridRNNTCTCWhisperLMModel instance
        copy_ctc_decoder: Whether to copy CTC decoder weights
    """
    # Extract student encoder weights (remove "student." prefix)
    print("\n=== Loading student encoder weights into RNNT encoder ===")
    student_encoder_state = {}
    for key, value in distill_state.items():
        if key.startswith("student."):
            new_key = key[len("student."):]
            student_encoder_state[new_key] = value
    print(f"Student encoder keys: {len(student_encoder_state)}")
    
    missing, unexpected = model.encoder.load_state_dict(student_encoder_state, strict=False)
    if missing:
        # Filter out slopes (expected to be missing, auto-computed for ALiBi)
        missing_non_slopes = [k for k in missing if 'slopes' not in k]
        if missing_non_slopes:
            print(f"  Missing keys (excluding slopes): {missing_non_slopes[:5]}...")
        else:
            print(f"  Missing keys: only slopes (auto-computed for ALiBi)")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
    print("  Encoder weights loaded successfully")
    
    # Extract and load CTC decoder weights if requested
    if copy_ctc_decoder:
        print("\n=== Loading CTC decoder weights ===")
        ctc_decoder_state = {}
        for key, value in distill_state.items():
            if key.startswith("ctc_decoder."):
                new_key = key[len("ctc_decoder."):]
                ctc_decoder_state[new_key] = value
        
        if ctc_decoder_state:
            print(f"CTC decoder keys: {len(ctc_decoder_state)}")
            missing, unexpected = model.ctc_decoder.load_state_dict(ctc_decoder_state, strict=False)
            if missing:
                print(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
            print("  CTC decoder weights loaded successfully")
        else:
            print("  No CTC decoder weights found in distillation checkpoint")


def load_qwen_decoder_weights(qwen_path, model):
    """
    Load Qwen decoder weights into RNNT model.
    
    Args:
        qwen_path: HuggingFace Qwen model path
        model: HybridRNNTCTCWhisperLMModel instance
    """
    print(f"\n=== Converting Qwen decoder ===")
    decoder_state, lm_head_weight = convert_qwen_decoder_weights(
        qwen_path,
        vocab_size=model.tokenizer.vocab_size,
        tie_weights=model.cfg.decoder.config.get("tie_word_embeddings", True)
    )
    
    missing, unexpected = model.decoder.prediction.load_state_dict(decoder_state, strict=False)
    if missing:
        print(f"  Decoder missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Decoder missing keys: {missing}")
    if unexpected:
        print(f"  Decoder unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Decoder unexpected keys: {unexpected}")
    print("  Decoder weights loaded successfully")
    
    # Load LLM head weights if not tied
    if lm_head_weight is not None and hasattr(model, 'llm_head') and hasattr(model.llm_head, 'decoder_layers'):
        model.llm_head.decoder_layers.weight.data.copy_(lm_head_weight)
        print("  LLM head weights loaded (untied)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CausalWhisperDistilModel checkpoint to HybridRNNTCTCWhisperLMModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input checkpoint
    parser.add_argument(
        "--distill-checkpoint",
        type=str,
        required=True,
        help="Path to distillation model .nemo checkpoint"
    )
    
    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to RNNT model config YAML"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for the RNNT .nemo checkpoint"
    )
    
    # Optional Qwen initialization
    parser.add_argument(
        "--qwen",
        type=str,
        required=True,
        help="HuggingFace Qwen model path for decoder initialization (optional)"
    )
    
    # CTC decoder option
    parser.add_argument(
        "--no-copy-ctc-decoder",
        action="store_true",
        help="Do not copy CTC decoder weights from distillation model"
    )
    
    args = parser.parse_args()
    
    copy_ctc_decoder = not args.no_copy_ctc_decoder
    
    print("=" * 60)
    print("Distillation to RNNT Checkpoint Conversion")
    print("=" * 60)
    print(f"Distill checkpoint: {args.distill_checkpoint}")
    print(f"RNNT config: {args.config}")
    print(f"Output: {args.output}")
    if args.qwen:
        print(f"Qwen: {args.qwen}")
    print(f"Copy CTC decoder: {copy_ctc_decoder}")
    
    # Load config and create dummy trainer
    config = OmegaConf.load(args.config)
    seed_everything(config.seed)
    dummy_trainer = pl.Trainer(**resolve_trainer_cfg(config.trainer))
    
    # Load distillation checkpoint
    print(f"\n=== Loading distillation checkpoint from {args.distill_checkpoint} ===")
    distill_state = load_nemo_checkpoint(args.distill_checkpoint)
    
    # Create RNNT model
    print(f"\n=== Creating HybridRNNTCTCWhisperLMModel ===")
    from src.models.rnnt_model import HybridRNNTCTCWhisperLMModel
    model = HybridRNNTCTCWhisperLMModel(cfg=config.model, trainer=dummy_trainer)
    
    # Load distillation student weights into encoder
    load_distill_weights_to_rnnt(distill_state, model, copy_ctc_decoder=copy_ctc_decoder)
    
    # Load Qwen decoder
    load_qwen_decoder_weights(args.qwen, model)
    
    # Save the model
    model.save_to(args.output)
    
    # Print summary
    encoder_total_params = sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in model.ctc_decoder.parameters())
    decoder_total_params = sum(p.numel() for p in model.decoder.parameters()) + sum(p.numel() for p in model.llm_head.parameters())
    joiner_total_params = sum(p.numel() for p in model.joint.parameters())
    
    print(f"\n=== Summary ===")
    print(f"  Encoder total parameters: {encoder_total_params:,}")
    print(f"  Decoder total parameters: {decoder_total_params:,}")
    print(f"  Joiner total parameters: {joiner_total_params:,}")
    print(f"  Total parameters: {encoder_total_params + decoder_total_params + joiner_total_params:,}")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
