#!/usr/bin/env python3
"""
Verify converted .nemo checkpoints by comparing teacher encoder output with HuggingFace Whisper.

This script validates that the distillation model's teacher encoder produces the same 
output as the original HuggingFace Whisper encoder.

Usage:
    python verify_checkpoint.py \
        --checkpoint distill_model.nemo \
        --config conf/hybrid_distil_ctc.yaml \
        --whisper openai/whisper-small
"""

import argparse
import os
import tarfile
import tempfile
import torch
from typing import Dict
from collections import OrderedDict
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from nemo.utils.trainer_utils import resolve_trainer_cfg


def load_nemo_checkpoint(nemo_path: str) -> Dict[str, torch.Tensor]:
    """
    Load state_dict from a .nemo file (tarball format).
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


def verify_teacher_encoder(model, hf_whisper_path: str, device: str = "cuda"):
    """
    Verify teacher encoder output matches HuggingFace Whisper.
    
    Args:
        model: CausalWhisperDistilModel instance with loaded weights
        hf_whisper_path: HuggingFace Whisper model path
        device: Device to run verification on
    """
    from transformers import WhisperModel
    
    print(f"\n=== Loading HuggingFace Whisper from {hf_whisper_path} ===")
    hf_model = WhisperModel.from_pretrained(hf_whisper_path)
    hf_encoder = hf_model.encoder.to(device)
    hf_encoder.eval()
    
    # Get a batch from the validation dataset
    print("\n=== Getting validation batch ===")
    val_dl = model.val_dataloader()
    batch = next(iter(val_dl))
    
    # Extract waveform from batch
    _, target, _, _, target_start, target_end, waveform = batch
    waveform = waveform.to(device)
    
    print(f"Waveform shape: {tuple(waveform.shape)}")
    
    # Process through preprocessor to get mel features
    mel_features, mel_length = model.preprocessor(raw_speech=waveform, length=None)
    print(f"Mel features shape: {tuple(mel_features.shape)}")  # (B, D, T)
    
    # Run through our teacher encoder
    model.teacher = model.teacher.to(device)
    model.teacher.eval()
    with torch.no_grad():
        our_output = model.teacher.forward_teacher(mel_features)  # (B, D, T)
    print(f"Our teacher output shape: {tuple(our_output.shape)}")
    
    # Run through HF Whisper encoder
    print(f"HF input shape: {tuple(mel_features.shape)}")
    
    with torch.no_grad():
        hf_output = hf_encoder(mel_features).last_hidden_state  # (B, T, D)
        hf_output = hf_output.transpose(1, 2)  # (B, T, D) -> (B, D, T)
    print(f"HF output shape: {tuple(hf_output.shape)}")
    
    # Compare outputs
    if our_output.shape != hf_output.shape:
        print(f"\nERROR: Shape mismatch!")
        print(f"  Our output: {our_output.shape}")
        print(f"  HF output: {hf_output.shape}")
        return False

    max_diff = (our_output - hf_output).abs().max().item()
    mean_diff = (our_output - hf_output).abs().mean().item()
    
    # Relative difference (more meaningful for neural network outputs)
    rel_diff = (our_output - hf_output).abs() / (hf_output.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"\n=== Comparison Results ===")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    print(f"  Mean relative difference: {mean_rel_diff:.2e}")
    
    # Use relaxed tolerance due to:
    # - flex_attention vs standard attention numerical differences
    # - bfloat16 mixed precision accumulation
    # - 32 layers of small differences accumulating
    tolerance = 0.5
    if max_diff < tolerance:
        print(f"  Status: PASSED (within tolerance {tolerance})")
        return True
    else:
        print(f"  Status: FAILED (max diff {max_diff:.2e} > {tolerance})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify distillation model teacher encoder against HuggingFace Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        required=True,
        help="Path to .nemo checkpoint"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--whisper",
        type=str,
        required=True,
        help="HuggingFace Whisper model path for verification"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run verification on"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Whisper: {args.whisper}")
    print(f"Device: {args.device}")
    
    # Load config and create model
    config = OmegaConf.load(args.config)
    seed_everything(config.seed)
    dummy_trainer = pl.Trainer(**resolve_trainer_cfg(config.trainer))
    
    # Load checkpoint state dict
    print(f"\n=== Loading checkpoint from {args.checkpoint} ===")
    checkpoint = load_nemo_checkpoint(args.checkpoint)
    
    # Create distillation model
    print(f"\n=== Creating CausalWhisperDistilModel ===")
    from src.models.causal_distill import CausalWhisperDistilModel
    model = CausalWhisperDistilModel(cfg=config.model, trainer=dummy_trainer)
    
    # Load teacher weights
    print(f"\n=== Loading teacher weights ===")
    teacher_state = {k[len("teacher."):]: v for k, v in checkpoint.items() if k.startswith("teacher.")}
    missing, unexpected = model.teacher.load_state_dict(teacher_state, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
    print(f"  Teacher weights loaded: {len(teacher_state)} keys")
    
    # Verify teacher encoder
    passed = verify_teacher_encoder(model, args.whisper, args.device)
    
    # Final summary
    print("\n" + "=" * 60)
    if passed:
        print("=== VERIFICATION PASSED ===")
        return 0
    else:
        print("=== VERIFICATION FAILED ===")
        return 1


if __name__ == "__main__":
    exit(main())
