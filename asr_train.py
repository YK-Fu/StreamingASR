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
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.cache_size_limit = 32
        typecheck.set_typecheck_enabled(False)  # prevent wrapt from blocking dynamo tracing

        enc_ckpt = cfg.model.get("encoder", {}).get("gradient_checkpointing", False)
        dec_ckpt = cfg.model.get("decoder", {}).get("gradient_checkpointing", False)
        if enc_ckpt or dec_ckpt:
            logging.warning(
                "torch_compile=True but gradient_checkpointing is enabled "
                f"(encoder={enc_ckpt}, decoder={dec_ckpt}). "
                "This reduces compile speedup from ~4.6x to ~3%. "
                "Set gradient_checkpointing: false in config for full benefit."
            )
        asr_model.encoder = torch.compile(
            asr_model.encoder, dynamic=True, fullgraph=False
        )
        asr_model.decoder = torch.compile(
            asr_model.decoder, dynamic=False, fullgraph=True
        )
        asr_model.ctc_decoder = torch.compile(
            asr_model.ctc_decoder, dynamic=True, fullgraph=False
        )
    gc.freeze()  # Prevent GC from scanning model objects in forked DataLoader workers (CoW mitigation)
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
