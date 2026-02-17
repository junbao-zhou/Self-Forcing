import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

machine_num = 8

gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
print(f"{MASTER_ADDR = }, {MASTER_PORT = }")

import sys
import time
from torch.distributed.run import main as torchrun_main
import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_id",
        type=str,
        help="The id for this training run, used for saving checkpoints and logs",
    )
    return parser


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    config_name = "self_forcing_dmd"
    sys.argv = [
        "torchrun",
        f"--nnodes={machine_num}",
        f"--nproc_per_node={gpu_num}",
        "--rdzv_id=5235",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={MASTER_ADDR}:{MASTER_PORT}",
        "train.py",
        "--",
        "--config-path=configs",
        f"--config-name={config_name}",
        f"logdir=logs/{config_name}-{args.save_id}",
        "disable_wandb=true",
    ]
    torchrun_main()
