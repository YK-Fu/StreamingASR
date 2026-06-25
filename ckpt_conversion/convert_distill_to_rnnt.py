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
from omegaconf import OmegaConf, open_dict

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
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            else:
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith(".ckpt"):
                            checkpoint = torch.load(os.path.join(root, f), map_location="cpu", weights_only=False)
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
        checkpoint = torch.load(nemo_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint


def convert_qwen_decoder_weights(
    hf_model_path: str,
    vocab_size: int,
) -> dict:
    """
    Convert HuggingFace Qwen to LLMDecoder.prediction state_dict.

    LLMDecoder wraps AutoModelForCausalLM (Qwen2ForCausalLM), so the HF state maps 1:1
    (keep "model." prefix + native "lm_head.weight"); the embedding and lm_head rows are
    trimmed to vocab_size. The lm_head<->embedding tie is handled by the model via
    config.tie_word_embeddings (lm_head.weight may be absent from a tied state_dict;
    strict=False at load time handles that).

    Args:
        hf_model_path: Path or HuggingFace model name
        vocab_size: Vocabulary size (HF Qwen vocab is usually larger; rows are trimmed)

    Returns:
        decoder_state_dict for model.decoder.prediction
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading Qwen model from {hf_model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, trust_remote_code=True)
    hf_state = hf_model.state_dict()

    converted = OrderedDict()
    for key, value in hf_state.items():
        if 'embed_tokens' in key or key == 'lm_head.weight':
            value = value[:vocab_size]
        converted[key] = value

    print(f"Qwen decoder conversion complete. Total keys: {len(converted)}")
    return converted


def _strip_orig_mod(key: str) -> str:
    """Strip torch.compile's '_orig_mod.' prefix anywhere in the key path.

    distiller_train.py wraps both student and teacher with torch.compile when
    `torch_compile: true` is set in the config — this rewrites their state_dict
    keys with a `_orig_mod.` infix at the wrap boundary (e.g.,
    `student._orig_mod.pre_encode.0.weight`). When transferring weights to the
    plain RNN-T encoder, that infix must be removed or load_state_dict treats
    all keys as missing/unexpected and leaves the encoder at random init.
    """
    return key.replace("_orig_mod.", "")


def load_distill_weights_to_rnnt(distill_state, model):
    """
    Load distillation model student weights into RNNT encoder, and ctc_decoder
    weights into the RNNT ctc_decoder.
    """
    # Extract student encoder weights (remove "student." prefix). Also strip any
    # `_orig_mod.` infix introduced by torch.compile wrapping during distill training.
    print("\n=== Loading student encoder weights into RNNT encoder ===")
    student_encoder_state = {}
    for key, value in distill_state.items():
        if key.startswith("student."):
            new_key = _strip_orig_mod(key[len("student."):])
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

    # Extract and load CTC decoder weights. Strip `_orig_mod.` for safety in case
    # ctc_decoder is also compiled in a future distill recipe.
    print("\n=== Loading CTC decoder weights ===")
    ctc_decoder_state = {}
    for key, value in distill_state.items():
        if key.startswith("ctc_decoder."):
            new_key = _strip_orig_mod(key[len("ctc_decoder."):])
            ctc_decoder_state[new_key] = value

    if ctc_decoder_state:
        print(f"CTC decoder keys: {len(ctc_decoder_state)}")
        # Filter out shape-mismatched keys before load_state_dict — strict=False handles
        # missing/unexpected keys but still raises on size mismatch for matched keys.
        # This makes the transfer robust across ProjHead shape changes (e.g. legacy
        # SimpleProj decoder_layers shape (V, enc_dim) → new ProjHead with hidden_dims
        # of a different size).
        target_state = model.ctc_decoder.state_dict()
        filtered_state = {}
        shape_mismatched = []
        for k, v in ctc_decoder_state.items():
            if k in target_state and target_state[k].shape != v.shape:
                shape_mismatched.append((k, tuple(v.shape), tuple(target_state[k].shape)))
            else:
                filtered_state[k] = v
        if shape_mismatched:
            print("  WARNING: shape-mismatched keys (kept at random init):")
            for k, src_shape, dst_shape in shape_mismatched[:5]:
                print(f"    {k}: src {src_shape} -> dst {dst_shape}")
        missing, unexpected = model.ctc_decoder.load_state_dict(filtered_state, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
        print(f"  CTC decoder weights loaded ({len(filtered_state)}/{len(ctc_decoder_state)} keys transferred)")
    else:
        print("  No CTC decoder weights found in distillation checkpoint")


def init_joint_from_ctc(model):
    """Initialize the joint from the CTC head, and zero-init project_prednet so
    the joint behaves approximately like CTC at step 0.

    With a ProjHead CTC head (hidden_dims=[H]: Linear(enc, H) -> SiLU ->
    Linear(H, V)) and a matching joint (project_encoder Linear + SiLU + final
    Linear), we copy:
        ctc_decoder.pre[0]          -> joint.enc                (when joint.enc is Linear)
        ctc_decoder.decoder_layers  -> joint final Linear
        joint.pred                  -> zeroed
    Plus zeroing the joint's biases so the joint output equals CTC's logits at
    step 0 (exact if joiner activation matches CTC's activation; SiLU on both).

    Falls back gracefully when CTC head is single-layer (hidden_dims=[], legacy
    SimpleProj) or when joint.enc is Identity (joint_hidden == enc_hidden).

    Additionally initializes the pruned-RNN-T simple projections so their
    prune-range scores start aligned with the trained heads instead of random:
        ctc_decoder (full state)    -> simple_am_proj           (identical ProjHead)
        decoder lm_head.weight      -> simple_lm_proj.decoder_layers.weight
    The acoustic side (simple_am_proj) mirrors the CTC head; the label side
    (simple_lm_proj) mirrors the Qwen LM head's next-token projection.
    """
    import torch.nn as nn

    # Locate the joint's final Linear (last Linear inside joint_net Sequential).
    joint_final = None
    for module in model.joint.joint_net:
        if isinstance(module, nn.Linear):
            joint_final = module
    if joint_final is None:
        raise RuntimeError("Could not find a Linear inside model.joint.joint_net")

    ctc_final = model.ctc_decoder.decoder_layers  # [V, H_ctc]
    if joint_final.weight.shape != ctc_final.weight.shape:
        raise RuntimeError(
            f"Shape mismatch: joint final Linear weight is {tuple(joint_final.weight.shape)} "
            f"but CTC final Linear weight is {tuple(ctc_final.weight.shape)}. For CTC-init, "
            f"set joint_hidden == ctc_decoder.hidden and ensure joint.num_classes+1 == "
            f"ctc.num_classes."
        )

    print("\n=== Initializing joint from CTC head ===")

    # Copy CTC final layer -> joint final Linear
    joint_final.weight.data.copy_(ctc_final.weight)
    if joint_final.bias is not None:
        joint_final.bias.data.zero_()
    print(f"  Copied CTC final weight {tuple(ctc_final.weight.shape)} -> joint final Linear; zeroed bias.")

    # Copy CTC first Linear (inside ProjHead.pre Sequential) -> joint.enc when both
    # are Linear. ProjHead with hidden_dims=[] has an empty `pre` Sequential and no
    # first Linear to copy (legacy single-layer CTC behavior).
    ctc_pre = getattr(model.ctc_decoder, 'pre', None)
    ctc_project = None
    if ctc_pre is not None:
        for module in ctc_pre:
            if isinstance(module, nn.Linear):
                ctc_project = module
                break
    if ctc_project is None:
        print(
            "  CTC head is single-layer (no `pre` Linear); nothing to copy into joint.enc. "
            "joint.enc left at its initialized values."
        )
    elif isinstance(model.joint.enc, nn.Identity):
        print(
            "  WARNING: joint.enc is Identity (joint_hidden == enc_hidden) but CTC has a "
            "project layer — CTC's first transform won't transfer. At step 0 the joint "
            "output will differ from CTC by approximately SiLU(project(x)) - SiLU(x). "
            "To enable exact transfer, set joint_hidden != enc_hidden so joint.enc is Linear."
        )
    elif model.joint.enc.weight.shape != ctc_project.weight.shape:
        print(
            f"  WARNING: joint.enc weight {tuple(model.joint.enc.weight.shape)} mismatches "
            f"ctc.project weight {tuple(ctc_project.weight.shape)} — skipping project copy."
        )
    else:
        model.joint.enc.weight.data.copy_(ctc_project.weight)
        if model.joint.enc.bias is not None:
            if ctc_project.bias is not None:
                model.joint.enc.bias.data.copy_(ctc_project.bias)
            else:
                model.joint.enc.bias.data.zero_()
        print(f"  Copied CTC project weight {tuple(ctc_project.weight.shape)} -> joint.enc.")

    # Zero project_prednet so the predictor contribution is null at step 0
    if isinstance(model.joint.pred, nn.Identity):
        print("  WARNING: project_prednet is Identity — cannot zero predictor contribution.")
    else:
        model.joint.pred.weight.data.zero_()
        if model.joint.pred.bias is not None:
            model.joint.pred.bias.data.zero_()
        print(f"  Zero-initialized project_prednet weight {tuple(model.joint.pred.weight.shape)} and bias.")

    joint_act = type(model.joint.joint_net[0]).__name__
    # Find the first non-Linear module inside ctc_decoder.pre — that's the activation
    # between the two CTC linears. None if pre is empty (single-layer CTC).
    ctc_act = 'Identity'
    if ctc_pre is not None:
        for module in ctc_pre:
            if not isinstance(module, nn.Linear):
                ctc_act = type(module).__name__
                break
    if joint_act == ctc_act:
        print(f"  Joint and CTC both use {joint_act} — CTC-init equivalence is exact at step 0.")
    else:
        print(
            f"  Note: joint activation is {joint_act}, CTC activation is {ctc_act}. "
            f"CTC-init equivalence is approximate."
        )

    # --- simple_am_proj <- CTC head ---
    # simple_am_proj and ctc_decoder are configured as the same ProjHead
    # (feat_in=encoder.d_model, hidden_dims=[joint_hidden]), so the CTC head's
    # state transfers 1:1. Shape-filter for robustness across config drift.
    simple_am = getattr(model, 'simple_am_proj', None)
    if simple_am is None:
        print("\n  simple_am_proj not present — skipping CTC-head copy.")
    else:
        print("\n=== Initializing simple_am_proj from CTC head ===")
        src_state = model.ctc_decoder.state_dict()
        tgt_state = simple_am.state_dict()
        filtered, mismatched = {}, []
        for k, v in src_state.items():
            if k in tgt_state and tgt_state[k].shape != v.shape:
                mismatched.append((k, tuple(v.shape), tuple(tgt_state[k].shape)))
            else:
                filtered[k] = v
        if mismatched:
            print("  WARNING: shape-mismatched keys (kept at random init):")
            for k, src_shape, dst_shape in mismatched[:5]:
                print(f"    {k}: src {src_shape} -> dst {dst_shape}")
        missing, unexpected = simple_am.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
        print(f"  simple_am_proj initialized from CTC head ({len(filtered)}/{len(src_state)} keys transferred).")

    # --- simple_lm_proj <- decoder LM head ---
    # simple_lm_proj is a single-Linear ProjHead (hidden_dims=[]): its
    # decoder_layers.weight is (vocab, hidden) and matches the Qwen lm_head
    # weight (both bias-free, rows already trimmed to the tokenizer vocab).
    simple_lm = getattr(model, 'simple_lm_proj', None)
    if simple_lm is None:
        print("\n  simple_lm_proj not present — skipping LM-head copy.")
    elif not hasattr(simple_lm, 'decoder_layers'):
        print("\n  simple_lm_proj has no decoder_layers (tie_weights?) — skipping LM-head copy.")
    else:
        print("\n=== Initializing simple_lm_proj from decoder LM head ===")
        lm_head_weight = model.decoder.prediction.lm_head.weight
        tgt_weight = simple_lm.decoder_layers.weight
        if tgt_weight.shape != lm_head_weight.shape:
            print(
                f"  WARNING: simple_lm_proj.decoder_layers weight {tuple(tgt_weight.shape)} "
                f"mismatches lm_head weight {tuple(lm_head_weight.shape)} — skipping copy."
            )
        else:
            tgt_weight.data.copy_(lm_head_weight)
            print(f"  Copied lm_head weight {tuple(lm_head_weight.shape)} -> simple_lm_proj.decoder_layers.")


def load_qwen_decoder_weights(qwen_path, model):
    """
    Load Qwen decoder weights into RNNT model.
    
    Args:
        qwen_path: HuggingFace Qwen model path
        model: HybridRNNTCTCWhisperLMModel instance
    """
    # Load decoder weights (incl. native lm_head) into decoder.prediction
    # (Qwen2ForCausalLM); the lm_head<->embedding tie is handled by the model via
    # config.tie_word_embeddings.
    print(f"\n=== Converting Qwen decoder (ForCausalLM) ===")
    decoder_state = convert_qwen_decoder_weights(
        qwen_path,
        vocab_size=model.tokenizer.vocab_size,
    )

    missing, unexpected = model.decoder.prediction.load_state_dict(decoder_state, strict=False)
    # `lm_head.weight` legitimately appears missing when the model ties it to embeddings.
    if missing:
        print(f"  Decoder missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Decoder missing keys: {missing}")
    if unexpected:
        print(f"  Decoder unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Decoder unexpected keys: {unexpected}")
    print("  Decoder + native lm_head weights loaded successfully")

    # --- Dedicated pruned-RNN-T simple projections: left at __init__ random init here.
    # simple_am_proj / simple_lm_proj are icefall-style; with --init-joint-from-ctc they
    # are subsequently initialized from the CTC head and the LM head respectively (see
    # init_joint_from_ctc, which runs after this). Otherwise they stay at the configured
    # simple_proj.init_scale random init and the loss-warmup phase trains them. ---
    for name in ("simple_am_proj", "simple_lm_proj"):
        if getattr(model, name, None) is not None:
            print(f"  {name}: kept at __init__ random init for now (use --init-joint-from-ctc to seed from CTC/LM head)")


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

    parser.add_argument(
        "--init-joint-from-ctc",
        action="store_true",
        help="Initialize the joint's final classifier from the CTC head and "
             "zero-init project_prednet. Requires joint_hidden == encoder.d_model "
             "so project_encoder is an Identity. Makes the RNN-T behave approximately "
             "like CTC at step 0 — a much better starting point than random init."
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("Distillation to RNNT Checkpoint Conversion")
    print("=" * 60)
    print(f"Distill checkpoint: {args.distill_checkpoint}")
    print(f"RNNT config: {args.config}")
    print(f"Output: {args.output}")
    if args.qwen:
        print(f"Qwen: {args.qwen}")

    # Load config and create dummy trainer
    config = OmegaConf.load(args.config)
    seed_everything(config.seed)
    with open_dict(config):
        config.trainer.devices = 1
        config.trainer.num_nodes = 1
        config.model.train_ds.manifest_filepath = []
        config.model.validation_ds.manifest_filepath = []
        config.model.test_ds.manifest_filepath = []
    dummy_trainer = pl.Trainer(**resolve_trainer_cfg(config.trainer))

    # Load distillation checkpoint
    print(f"\n=== Loading distillation checkpoint from {args.distill_checkpoint} ===")
    distill_state = load_nemo_checkpoint(args.distill_checkpoint)

    # Create RNNT model
    print(f"\n=== Creating HybridRNNTCTCWhisperLMModel ===")
    from src.models.rnnt_model import HybridRNNTCTCWhisperLMModel
    model = HybridRNNTCTCWhisperLMModel(cfg=config.model, trainer=dummy_trainer)

    # Load distillation student weights into encoder and ctc_decoder
    load_distill_weights_to_rnnt(distill_state, model)

    # Load Qwen decoder
    load_qwen_decoder_weights(args.qwen, model)

    # Optionally initialize joint from CTC head (must happen AFTER ctc_decoder
    # is loaded; it reads ctc_decoder.decoder_layers.weight).
    if args.init_joint_from_ctc:
        init_joint_from_ctc(model)

    # Save the model
    model.save_to(args.output)
    
    # Print summary
    encoder_total_params = sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in model.ctc_decoder.parameters()) + sum(p.numel() for p in model.simple_am_proj.parameters())
    decoder_total_params = sum(p.numel() for p in model.decoder.parameters()) + sum(p.numel() for p in model.simple_lm_proj.parameters())
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
