import os
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
import torch
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

import bitsandbytes as bnb
from nemo.core.optim import register_optimizer
from nemo.core.config.optimizers import AdamWParams
register_optimizer("adamw_8bit", bnb.optim.AdamW8bit, AdamWParams)

from src.models.rnnt_model import HybridRNNTCTCWhisperLMModel

@hydra_runner(config_path="./conf", config_name="hybrid_transducer_ctc")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    seed_everything(cfg.seed)
    # Enable TF32 for fp32 matmul ops (encoder/decoder ops are bf16-mixed via autocast,
    # but on_after_backward gradient norms, NLLLoss reductions, and k2's surrounding fp32
    # paths still benefit). Verified safe with k2 — its CUDA kernels bypass cuBLAS so
    # TF32 doesn't alter the recursion. See /tmp/k2_tf32_test.py.
    torch.set_float32_matmul_precision('high')
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    trainer = pl.Trainer(**trainer_cfg)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = HybridRNNTCTCWhisperLMModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    if cfg.get("torch_compile", False):
        from nemo.core.classes.common import typecheck
        torch._dynamo.config.suppress_errors = True  # fall back to eager on any unhandled break
        # 6 audio shapes (5s-step chunks) × encoder + decoder dynamic graph + ctc/llm/joint
        # dynamic graphs + val-time shapes. 20 is conservative headroom; dynamo will warn
        # before exceeding.
        torch._dynamo.config.cache_size_limit = 20
        typecheck.set_typecheck_enabled(False)  # prevent wrapt from blocking dynamo tracing
        # Compile strategy:
        #   - encoder: dynamic=False, max-autotune-no-cudagraphs → 6 per-shape specialized
        #     compiles (one per audio_chunk_step bucket). Each Triton kernel is autotuned
        #     for its specific shape. Cudagraphs disabled because they require static
        #     shapes across the WHOLE step, which we can't guarantee (text U varies).
        #   - decoder: dynamic=True, max-autotune-no-cudagraphs → one shape-polymorphic
        #     graph handling all text lengths. Inductor still picks decent kernels per
        #     runtime shape; we just lose constant-folding on U.
        #   - ctc_decoder / llm_head / joint.joint_after_projection: dynamic=True with
        #     default mode. These are short fusion chains; max-autotune costs more compile
        #     time than it saves at runtime for such small graphs.
        # k2 RNNT loss ops sit outside all of these compile boundaries (uncompilable —
        # custom autograd functions with non-traceable CUDA kernels).
        enc_ckpt = cfg.model.get("encoder", {}).get("gradient_checkpointing", False)
        dec_ckpt = cfg.model.get("decoder", {}).get("gradient_checkpointing", False)
        if enc_ckpt or dec_ckpt:
            logging.warning(
                "torch_compile=True but gradient_checkpointing is enabled "
                f"(encoder={enc_ckpt}, decoder={dec_ckpt}). "
                "This reduces compile speedup from ~4.6x to ~3%. "
                "Set gradient_checkpointing: false in config for full benefit."
            )
        logging.info(
            "torch.compile enabled: encoder (dynamic=False, max-autotune-no-cudagraphs), "
            "decoder (dynamic=True, max-autotune-no-cudagraphs), "
            "ctc_decoder/llm_head/joint.joint_after_projection (dynamic=True)"
        )
        asr_model.encoder = torch.compile(
            asr_model.encoder, dynamic=False, fullgraph=False
        )
        asr_model.decoder = torch.compile(
            asr_model.decoder, dynamic=True, fullgraph=False, mode='max-autotune-no-cudagraphs'
        )
        asr_model.ctc_decoder = torch.compile(
            asr_model.ctc_decoder, dynamic=False, fullgraph=True
        )
        # Compile only joint_after_projection — the surrounding forward_fused_loss
        # contains k2 calls that dynamo can't trace. joint_after_projection captures
        # the broadcast-add + SiLU + Dropout + Linear + log_softmax fusion.
        asr_model.joint.joint_after_projection = torch.compile(
            asr_model.joint.joint_after_projection, dynamic=False, fullgraph=True
        )

    gc.freeze()  # Prevent GC from scanning model objects in forked DataLoader workers (CoW mitigation)
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
