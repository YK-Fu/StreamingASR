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
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    trainer = pl.Trainer(**trainer_cfg)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = CausalWhisperDistilModel(cfg=cfg.model, trainer=trainer)
    
    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    if cfg.get("torch_compile", False):
        import torch._dynamo
        from nemo.core.classes.common import typecheck
        torch._dynamo.config.suppress_errors = True
        typecheck.set_typecheck_enabled(False)  # prevent wrapt from blocking dynamo tracing
        # No k2 in distillation path and input shape is fixed — use static compilation for
        # maximum kernel specialization. fullgraph=False required because flex_attention's
        # create_block_mask calls inspect.signature on a closure (alibi_mask_mod), which
        # dynamo cannot trace; it graph-breaks there and compiles the surrounding ops.
        # drop_last=True in the dataloader is required to avoid a shape recompilation on
        # the last (smaller) batch of each epoch.
        logging.info("torch.compile enabled: compiling student and teacher encoders (dynamic=False, fullgraph=False)")
        asr_model.student = torch.compile(asr_model.student, dynamic=False, fullgraph=False)
        asr_model.teacher = torch.compile(asr_model.teacher, dynamic=False, fullgraph=False)

    gc.freeze()  # Prevent GC from scanning model objects in forked DataLoader workers (CoW mitigation)
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
