import lightning.pytorch as pl
from lightning.pytorch import seed_everything
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

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

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter