import time
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from trainer import (
    DiffusionTrainer,
    GANTrainer,
    ODETrainer,
    ScoreDistillationTrainer,
)

import logging
from pathlib import Path
from utils.misc import format_dict
from utils.logging import (
    _current_node_rank,
    _current_process_rank,
    _configure_logging,
)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="rolling_forcing_dmd",
)
def main(
    config: DictConfig,
):
    # Strict mode: CLI override keys must already exist in cfg, otherwise error.
    OmegaConf.set_struct(config, True)

    config_name = HydraConfig.get().job.config_name

    # Keep paths stable even if Hydra changes working dir
    orig_cwd = hydra.utils.get_original_cwd()

    logdir = Path(config.logdir)
    if not logdir.is_absolute():
        logdir = Path(orig_cwd) / logdir
    logdir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    _configure_logging(
        logdir / f"train-node{_current_node_rank()}-rank{_current_process_rank()}-{time_str}.log"
    )

    logging.info(
        f"""
{config_name = }
config = {format_dict(config)}
"""
    )

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
