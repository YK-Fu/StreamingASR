#!/usr/bin/env python3
"""
Convert HuggingFace Whisper encoder and Qwen decoder checkpoints to NeMo format.

This script converts:
- Whisper encoder -> WhisperEncoder (for distillation teacher or custom encoder)
- Qwen model -> LLMDecoder (for RNNT predictor)

Supports word embedding tie/untie for Qwen.

The script instantiates the actual NeMo models and uses model.save_to() to create
proper .nemo files (tarball with config + weights).
"""

import argparse
import torch
from typing import Dict, Optional
from collections import OrderedDict
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from nemo.utils.trainer_utils import resolve_trainer_cfg


def validate_whisper_config(hf_model_path: str, encoder_cfg) -> None:
    """
    Validate that the NeMo encoder config matches the HuggingFace Whisper model.
    
    Raises ValueError if there's a mismatch in critical parameters.
    """
    from transformers import WhisperConfig
    
    print(f"\n=== Validating Whisper config ===")
    hf_config = WhisperConfig.from_pretrained(hf_model_path)
    
    # Get values from NeMo config
    nemo_d_model = encoder_cfg.get("d_model")
    nemo_n_heads = encoder_cfg.get("n_heads")
    nemo_n_layers = encoder_cfg.get("n_layers")
    nemo_ff_expansion = encoder_cfg.get("ff_expansion_factor", 4)
    
    # HF Whisper values
    hf_d_model = hf_config.d_model
    hf_n_heads = hf_config.encoder_attention_heads
    hf_n_layers = hf_config.encoder_layers
    hf_ff_dim = hf_config.encoder_ffn_dim
    hf_ff_expansion = hf_ff_dim // hf_d_model
    
    print(f"  HuggingFace Whisper: d_model={hf_d_model}, n_heads={hf_n_heads}, n_layers={hf_n_layers}, ff_expansion={hf_ff_expansion}")
    print(f"  NeMo config:         d_model={nemo_d_model}, n_heads={nemo_n_heads}, n_layers={nemo_n_layers}, ff_expansion={nemo_ff_expansion}")
    
    errors = []
    
    if nemo_d_model != hf_d_model:
        errors.append(f"d_model mismatch: config has {nemo_d_model}, but HF Whisper has {hf_d_model}")
    
    if nemo_n_heads != hf_n_heads:
        errors.append(f"n_heads mismatch: config has {nemo_n_heads}, but HF Whisper has {hf_n_heads}")
    
    if nemo_n_layers != hf_n_layers:
        errors.append(f"n_layers mismatch: config has {nemo_n_layers}, but HF Whisper has {hf_n_layers}")
    
    if nemo_ff_expansion != hf_ff_expansion:
        errors.append(f"ff_expansion_factor mismatch: config has {nemo_ff_expansion}, but HF Whisper has {hf_ff_expansion}")
    
    if errors:
        error_msg = "Config validation failed:\n  " + "\n  ".join(errors)
        error_msg += f"\n\nPlease update your config to match the HF Whisper model '{hf_model_path}'."
        raise ValueError(error_msg)
    
    print(f"  Config validation passed!")


def convert_whisper_encoder_weights(
    hf_model_path: str,
    include_position_embeddings: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace Whisper encoder to WhisperEncoder state_dict (no prefix).
    
    Architecture mapping:
        HF Whisper                                  -> WhisperEncoder
        ----------------------------------------------------------------
        encoder.conv1.weight/bias                   -> pre_encode.0.weight/bias
        encoder.conv2.weight/bias                   -> pre_encode.2.weight/bias
        encoder.embed_positions.weight              -> pos_enc.weight
        encoder.layers[i].self_attn.{q,k,v}_proj    -> layers[i].self_attn.linear_qkv (fused)
        encoder.layers[i].self_attn.out_proj        -> layers[i].self_attn.linear_out
        encoder.layers[i].self_attn_layer_norm      -> layers[i].norm_self_att
        encoder.layers[i].fc1                       -> layers[i].feed_forward.linear1
        encoder.layers[i].fc2                       -> layers[i].feed_forward.linear2
        encoder.layers[i].final_layer_norm          -> layers[i].norm_feed_forward
        encoder.layer_norm                          -> layer_norm
    
    Args:
        hf_model_path: Path or HuggingFace model name (e.g., "openai/whisper-small")
        include_position_embeddings: Whether to include learned position embeddings
    
    Returns:
        Converted state_dict with proper keys for WhisperEncoder (no prefix)
    """
    from transformers import WhisperModel
    
    print(f"Loading Whisper model from {hf_model_path}...")
    hf_model = WhisperModel.from_pretrained(hf_model_path)
    hf_state = hf_model.encoder.state_dict()
    
    converted = OrderedDict()
    
    # Conv layers (subsampling)
    # HF: conv1, conv2 -> Ours: pre_encode.0, pre_encode.2
    converted["pre_encode.0.weight"] = hf_state["conv1.weight"]
    converted["pre_encode.0.bias"] = hf_state["conv1.bias"]
    converted["pre_encode.2.weight"] = hf_state["conv2.weight"]
    converted["pre_encode.2.bias"] = hf_state["conv2.bias"]
    
    # Position embeddings (optional, only for "learned" mode)
    if include_position_embeddings and "embed_positions.weight" in hf_state:
        converted["pos_enc.weight"] = hf_state["embed_positions.weight"]
    
    # Transformer layers
    num_layers = len([k for k in hf_state.keys() if k.startswith("layers.") and ".self_attn.q_proj.weight" in k])
    print(f"Converting {num_layers} transformer layers...")
    
    for i in range(num_layers):
        layer_prefix = f"layers.{i}"
        hf_layer_prefix = f"layers.{i}"
        
        # Fuse Q, K, V projections into single linear_qkv
        q_weight = hf_state[f"{hf_layer_prefix}.self_attn.q_proj.weight"]
        k_weight = hf_state[f"{hf_layer_prefix}.self_attn.k_proj.weight"]
        v_weight = hf_state[f"{hf_layer_prefix}.self_attn.v_proj.weight"]
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        converted[f"{layer_prefix}.self_attn.linear_qkv.weight"] = qkv_weight
        
        # Fuse Q, K, V biases if they exist
        if f"{hf_layer_prefix}.self_attn.q_proj.bias" in hf_state:
            q_bias = hf_state[f"{hf_layer_prefix}.self_attn.q_proj.bias"]
            k_bias = torch.zeros_like(q_bias)       # There is no bias in K projection in Whisper
            v_bias = hf_state[f"{hf_layer_prefix}.self_attn.v_proj.bias"]
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            converted[f"{layer_prefix}.self_attn.linear_qkv.bias"] = qkv_bias
        
        # Output projection
        converted[f"{layer_prefix}.self_attn.linear_out.weight"] = hf_state[f"{hf_layer_prefix}.self_attn.out_proj.weight"]
        if f"{hf_layer_prefix}.self_attn.out_proj.bias" in hf_state:
            converted[f"{layer_prefix}.self_attn.linear_out.bias"] = hf_state[f"{hf_layer_prefix}.self_attn.out_proj.bias"]
        
        # Self-attention layer norm -> norm_self_att
        converted[f"{layer_prefix}.norm_self_att.weight"] = hf_state[f"{hf_layer_prefix}.self_attn_layer_norm.weight"]
        converted[f"{layer_prefix}.norm_self_att.bias"] = hf_state[f"{hf_layer_prefix}.self_attn_layer_norm.bias"]
        
        # Feed forward layers
        converted[f"{layer_prefix}.feed_forward.linear1.weight"] = hf_state[f"{hf_layer_prefix}.fc1.weight"]
        converted[f"{layer_prefix}.feed_forward.linear1.bias"] = hf_state[f"{hf_layer_prefix}.fc1.bias"]
        converted[f"{layer_prefix}.feed_forward.linear2.weight"] = hf_state[f"{hf_layer_prefix}.fc2.weight"]
        converted[f"{layer_prefix}.feed_forward.linear2.bias"] = hf_state[f"{hf_layer_prefix}.fc2.bias"]
        
        # Final layer norm -> norm_feed_forward
        converted[f"{layer_prefix}.norm_feed_forward.weight"] = hf_state[f"{hf_layer_prefix}.final_layer_norm.weight"]
        converted[f"{layer_prefix}.norm_feed_forward.bias"] = hf_state[f"{hf_layer_prefix}.final_layer_norm.bias"]
    
    # Final layer norm
    converted["layer_norm.weight"] = hf_state["layer_norm.weight"]
    converted["layer_norm.bias"] = hf_state["layer_norm.bias"]
    
    print(f"Whisper encoder conversion complete. Total keys: {len(converted)}")
    return converted


def convert_qwen_decoder_weights(
    hf_model_path: str,
    vocab_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace Qwen to LLMDecoder.prediction state_dict.

    LLMDecoder now wraps AutoModelForCausalLM (Qwen2ForCausalLM), so the HF state_dict
    maps 1:1 — keep the "model." prefix and the native "lm_head.weight". The embedding
    and lm_head rows are trimmed to vocab_size. The lm_head<->embedding tie is handled
    by the model itself via config.tie_word_embeddings (for tied models lm_head.weight
    may be absent from the state_dict; strict=False at load time handles that).

    Architecture mapping:
        HF Qwen2ForCausalLM          -> LLMDecoder.prediction (Qwen2ForCausalLM)
        ----------------------------------------------------------------
        model.embed_tokens.weight    -> model.embed_tokens.weight (trimmed)
        model.layers[i].*            -> model.layers[i].*
        model.norm.weight            -> model.norm.weight
        lm_head.weight               -> lm_head.weight (trimmed; absent if tied)

    Args:
        hf_model_path: Path or HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
        vocab_size: Vocabulary size (the HF Qwen vocab is usually larger; rows are trimmed)

    Returns:
        decoder_state_dict for model.decoder.prediction
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading Qwen model from {hf_model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, trust_remote_code=True)
    hf_state = hf_model.state_dict()

    converted = OrderedDict()
    for key, value in hf_state.items():
        # Trim the vocab dimension of the embedding and the (untied) lm_head.
        if 'embed_tokens' in key or key == 'lm_head.weight':
            value = value[:vocab_size]
        converted[key] = value

    print(f"Qwen decoder conversion complete. Total keys: {len(converted)}")
    return converted


def create_rnnt_model_checkpoint(whisper_path, qwen_path, model):
    # Validate config before conversion
    validate_whisper_config(whisper_path, model.cfg.encoder)

    # Convert and load encoder weights
    print(f"\n=== Converting Whisper encoder ===")
    encoder_state = convert_whisper_encoder_weights(
        whisper_path,
        include_position_embeddings=model.cfg.encoder.get("position_embedding_type", "alibi") == "learned"
    )

    # Load encoder weights
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"  Encoder missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Encoder missing keys: {missing}")
    if unexpected:
        print(f"  Encoder unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Encoder unexpected keys: {unexpected}")
    print(f"  Encoder weights loaded successfully")
    
    # Convert and load decoder weights (incl. native lm_head) into decoder.prediction
    # (Qwen2ForCausalLM). The lm_head<->embedding tie is handled by the model via
    # config.tie_word_embeddings.
    print(f"\n=== Converting Qwen decoder (ForCausalLM) ===")
    decoder_state = convert_qwen_decoder_weights(
        qwen_path,
        vocab_size=model.tokenizer.vocab_size,
    )

    missing, unexpected = model.decoder.prediction.load_state_dict(decoder_state, strict=False)
    # `lm_head.weight` legitimately appears as a missing key when the model ties it to
    # the embedding (tie_word_embeddings=True) — it shares embed_tokens.weight.
    if missing:
        print(f"  Decoder missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Decoder missing keys: {missing}")
    if unexpected:
        print(f"  Decoder unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Decoder unexpected keys: {unexpected}")
    print(f"  Decoder + native lm_head weights loaded successfully")

    # --- Dedicated pruned-RNN-T simple projections: kept at RANDOM init ---
    # simple_am_proj / simple_lm_proj (icefall-style, untied from the CTC and LM heads)
    # are already randomly initialized by the model __init__ using the configured
    # simple_proj.init_scale. They are NOT loaded from Qwen.
    for name in ("simple_am_proj", "simple_lm_proj"):
        if getattr(model, name, None) is not None:
            print(f"  {name}: kept at __init__ random init (config simple_proj.init_scale); NOT loaded from Qwen")


def create_distill_model_checkpoint(whisper_path, model):
    # Validate teacher config before conversion
    validate_whisper_config(whisper_path, model.cfg.teacher)
    
    # Convert and load teacher encoder weights (with position embeddings)
    print(f"\n=== Converting Whisper encoder for teacher ===")
    teacher_state = convert_whisper_encoder_weights(
        whisper_path,
        include_position_embeddings=True  # Teacher uses learned position embeddings
    )
    
    missing, unexpected = model.teacher.load_state_dict(teacher_state, strict=False)
    if missing:
        print(f"  Teacher missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Teacher missing keys: {missing}")
    if unexpected:
        print(f"  Teacher unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Teacher unexpected keys: {unexpected}")
    print(f"  Teacher weights loaded successfully")

    # Convert and load student encoder weights (without position embeddings - ALiBi)
    print(f"\n=== Converting Whisper encoder for student ===")
    student_state = convert_whisper_encoder_weights(
        whisper_path,
        include_position_embeddings=False  # Student uses ALiBi
    )
    
    missing, unexpected = model.student.load_state_dict(student_state, strict=False)
    if missing:
        print(f"  Student missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Student missing keys: {missing}")
    if unexpected:
        print(f"  Student unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Student unexpected keys: {unexpected}")
    print(f"  Student weights loaded successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Whisper/Qwen checkpoints to NeMo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model source arguments
    parser.add_argument(
        "--whisper",
        type=str,
        default=None,
        help="HuggingFace Whisper model path or name (e.g., openai/whisper-small)"
    )
    parser.add_argument(
        "--qwen",
        type=str,
        default=None,
        help="HuggingFace Qwen model path or name (e.g., Qwen/Qwen2.5-0.5B)"
    )
    
    # Config file (required for proper NeMo format)
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., conf/hybrid_transducer_ctc.yaml)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for the .nemo checkpoint"
    )
    
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    seed_everything(config.seed)
    dummy_trainer = pl.Trainer(**resolve_trainer_cfg(config.trainer))
    # Validate arguments
    if config.model.get("teacher", None) is not None:
        if args.whisper is None:
            parser.error("--whisper is required for distillation")

        from src.models.causal_distill import CausalWhisperDistilModel
        model = CausalWhisperDistilModel(cfg=config.model, trainer=dummy_trainer)
        create_distill_model_checkpoint(
            whisper_path=args.whisper,
            model=model
        )
        teacher_total_params = sum(p.numel() for p in model.teacher.parameters())
        student_total_params = sum(p.numel() for p in model.student.parameters()) + sum(p.numel() for p in model.ctc_decoder.parameters())
        print(f"\n=== Summary ===")
        print(f"  Teacher total parameters: {teacher_total_params:,}")
        print(f"  Student total parameters: {student_total_params:,}")
        
    else:
        if args.whisper is None:
            parser.error("--whisper is required for rnnt model")
        if args.qwen is None:
            parser.error("--qwen is required for rnnt model")

        from src.models.rnnt_model import HybridRNNTCTCWhisperLMModel
        model = HybridRNNTCTCWhisperLMModel(cfg=config.model, trainer=dummy_trainer)
        create_rnnt_model_checkpoint(
            whisper_path=args.whisper,
            qwen_path=args.qwen,
            model=model
        )
        encoder_total_params = sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in model.ctc_decoder.parameters()) + sum(p.numel() for p in model.simple_am_proj.parameters())
        decoder_total_params = sum(p.numel() for p in model.decoder.parameters()) + sum(p.numel() for p in model.simple_lm_proj.parameters())
        joiner_total_params = sum(p.numel() for p in model.joint.parameters())
        print(f"\n=== Summary ===")
        print(f"  Encoder total parameters: {encoder_total_params:,}")
        print(f"  Decoder total parameters: {decoder_total_params:,}")
        print(f"  Joiner total parameters: {joiner_total_params:,}")
    model.save_to(args.output)
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
