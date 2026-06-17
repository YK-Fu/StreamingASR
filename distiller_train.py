import os
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
import bitsandbytes as bnb

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.core.optim import register_optimizer
from nemo.core.config.optimizers import AdamWParams
register_optimizer("adamw_8bit", bnb.optim.AdamW8bit, AdamWParams)

from src.models.causal_distill import CausalWhisperDistilModel

@hydra_runner(config_path="./conf", config_name="hybrid_distil_ctc")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    seed_everything(cfg.seed)
    # Enable TF32 for fp32 matmul ops (most ops are bf16-mixed via autocast, but
    # on_after_backward gradient norms and reduction paths still benefit).
    torch.set_float32_matmul_precision('high')
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    trainer = pl.Trainer(**trainer_cfg)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = CausalWhisperDistilModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    if cfg.get("torch_compile", False):
        from nemo.core.classes.common import typecheck
        torch._dynamo.config.suppress_errors = True
        # Static distill shapes (audio_chunk_step=30, drop_last=True) → just a handful
        # of graphs (student + teacher + ctc_decoder). 20 gives headroom for val-time
        # shapes without warnings.
        torch._dynamo.config.cache_size_limit = 20
        typecheck.set_typecheck_enabled(False)  # prevent wrapt from blocking dynamo tracing
        # No k2 in distillation path and input shape is fixed — use static compilation for
        # maximum kernel specialization. fullgraph=False required because flex_attention's
        # create_block_mask calls inspect.signature on a closure (alibi_mask_mod), which
        # dynamo cannot trace; it graph-breaks there and compiles the surrounding ops.
        # drop_last=True in the dataloader is required to avoid a shape recompilation on
        # the last (smaller) batch of each epoch.
        logging.info(
            "torch.compile enabled: student/teacher (dynamic=False, max-autotune-no-cudagraphs), "
            "ctc_decoder (dynamic=False)"
        )
        asr_model.student = torch.compile(
            asr_model.student, dynamic=False, fullgraph=False
        )
        asr_model.teacher = torch.compile(
            asr_model.teacher, dynamic=False, fullgraph=False
        )
        # ctc_decoder runs on the static (B, 1280, 750) student output; dynamic=False
        # gives a fully specialized fused kernel for Linear→SiLU→Linear→log_softmax.
        asr_model.ctc_decoder = torch.compile(
            asr_model.ctc_decoder, dynamic=False, fullgraph=True
        )

    gc.freeze()  # Prevent GC from scanning model objects in forked DataLoader workers (CoW mitigation)
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
