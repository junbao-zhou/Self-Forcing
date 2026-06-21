import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
print(f"{MASTER_ADDR = }, {MASTER_PORT = }")

import sys
import shlex
from torch.distributed.run import main as torchrun_main
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainSettings(BaseSettings):
    """Parse train launcher settings from command-line arguments."""

    model_config = SettingsConfigDict(cli_parse_args=True)

    save_id: str = Field(
        default="test",
        description="The id for this training run, used for saving checkpoints and logs.",
    )
    machine_num: int = Field(
        default=1,
        description="Number of machines to use for distributed training.",
    )
    config_name: str = Field(
        default="self_forcing_dmd",
        description="Hydra config name.",
    )
    train_extra_arguments: str = Field(
        default="",
        description="Extra arguments passed to train.py after built-in Hydra overrides.",
    )


def parse_train_extra_arguments(train_extra_arguments: str) -> list[str]:
    """Parse a shell-like string into extra train.py arguments."""

    return shlex.split(train_extra_arguments)


if __name__ == "__main__":
    settings = TrainSettings()
    train_extra_arguments = parse_train_extra_arguments(settings.train_extra_arguments)
    sys.argv = [
        "torchrun",
        f"--nnodes={settings.machine_num}",
        f"--nproc_per_node={gpu_num}",
        f"--rdzv_id={settings.save_id}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={MASTER_ADDR}:{MASTER_PORT}",
        "train.py",
        "--",
        "--config-path=configs",
        f"--config-name={settings.config_name}",
        f"logdir=logs/{settings.save_id}-{settings.config_name}",
        "disable_wandb=true",
        *train_extra_arguments,
    ]
    torchrun_main()
