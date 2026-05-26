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
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    trainer = pl.Trainer(**trainer_cfg)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = HybridRNNTCTCWhisperLMModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    if cfg.get("torch_compile", False):
        import torch._dynamo
        from nemo.core.classes.common import typecheck
        torch._dynamo.config.suppress_errors = True  # fall back to eager on any unhandled break
        torch._dynamo.config.cache_size_limit = 16  # 6 training shapes (5s-step chunks) + val shapes
        typecheck.set_typecheck_enabled(False)  # prevent wrapt from blocking dynamo tracing
        # Compile encoder and decoder — k2 RNNT joint uses a custom autograd.Function
        # that cannot be traced; the surrounding ops still benefit from compilation.
        # dynamic=True handles variable-length audio without per-shape recompilation.
        # NOTE: gradient_checkpointing on encoder/decoder reduces the compile benefit from
        # ~4.6x to ~3% (FlexAttention fused kernels cannot fire during checkpointed forward).
        # Disable checkpointing in config for maximum throughput; re-enable if GPU OOM.
        enc_ckpt = cfg.model.get("encoder", {}).get("gradient_checkpointing", False)
        dec_ckpt = cfg.model.get("decoder", {}).get("gradient_checkpointing", False)
        if enc_ckpt or dec_ckpt:
            logging.warning(
                "torch_compile=True but gradient_checkpointing is enabled "
                f"(encoder={enc_ckpt}, decoder={dec_ckpt}). "
                "This reduces compile speedup from ~4.6x to ~3%. "
                "Set gradient_checkpointing: false in config for full benefit."
            )
        logging.info("torch.compile enabled: compiling encoder and decoder (dynamic=True)")
        asr_model.encoder = torch.compile(asr_model.encoder, dynamic=True, fullgraph=False)
        asr_model.decoder = torch.compile(asr_model.decoder, dynamic=True, fullgraph=False)

    gc.freeze()  # Prevent GC from scanning model objects in forked DataLoader workers (CoW mitigation)
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter