import gc
import logging
from utils.logging import logger
from functools import partial
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.distributed import (
    fsdp_state_dict,
    launch_distributed_job,
)
from utils.misc import set_seed


def seed_dataloader_worker(
    worker_id: int,
    training_seed: int,
) -> None:
    worker_seed = training_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()

        self.device = torch.cuda.current_device()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = self.global_rank == 0
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.model_init_seed = int(config.seed)
        self.training_seed = self.model_init_seed + self.global_rank

        set_seed(self.model_init_seed)

        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=os.path.join(config.logdir, "tensorboard"))
        else:
            self.writer = None

        self.output_path = config.logdir
        self.max_grad_norm = 10.0
        self.previous_time = None
        self.causal = config.causal

    def set_training_seed(self) -> None:
        set_seed(self.training_seed)

    def build_distributed_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        num_workers: int = 8,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.distributed.DistributedSampler,
    ]:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=self.model_init_seed,
        )

        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(self.training_seed)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=partial(
                seed_dataloader_worker,
                training_seed=self.training_seed,
            ),
            generator=data_loader_generator,
        )
        return dataloader, sampler

    def save_checkpoint(self, state_dict):
        if self.is_main_process:
            os.makedirs(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"),
                exist_ok=True,
            )
            torch.save(
                state_dict,
                os.path.join(
                    self.output_path,
                    f"checkpoint_model_{self.step:06d}",
                    "model.pt",
                ),
            )
            logger.info(
                f"Model saved to {os.path.join(self.output_path, f'checkpoint_model_{self.step:06d}', 'model.pt')}"
            )

            max_checkpoints = self.config.max_checkpoints
            checkpoints = sorted(
                [d for d in os.listdir(self.output_path) if d.startswith("checkpoint_model_")]
            )
            if len(checkpoints) > max_checkpoints:
                for old_ckpt in checkpoints[:-max_checkpoints]:
                    shutil.rmtree(os.path.join(self.output_path, old_ckpt))
                    logger.info(f"Deleted old checkpoint: {old_ckpt}")

    def run_inference(self, generator):
        logger.info("Gathering generator state dict for inference...")
        generator_state = fsdp_state_dict(generator)

        if not self.is_main_process:
            return

        with open("prompts/MovieGenVideoBench_extended.txt", "r") as f:
            prompts = [line.strip() for line in f.readlines()[0 : self.config.num_inference_prompts * 2 : 2]]

        pipeline = CausalInferencePipeline(
            self.config,
            device=self.device,
            vae_offload_cpu=getattr(self.config, "vae_offload_cpu", False),
        )
        pipeline.generator.load_state_dict(generator_state)
        pipeline = pipeline.to(dtype=torch.bfloat16)

        with torch.no_grad():
            for idx, prompt in enumerate(prompts):
                noise = torch.randn([1, 21, 16, 60, 104], device=self.device, dtype=torch.bfloat16)
                previous_level = logger.level
                logger.setLevel(max(previous_level, logging.WARNING))
                video, _ = pipeline.inference(
                    noise=noise, text_prompts=[prompt], return_latents=True
                )
                logger.setLevel(previous_level)
                video = rearrange(video, "b t c h w -> b t h w c").cpu()
                video = 255.0 * video
                output_video_dir = Path(self.output_path) / f"inference_videos_step{self.step:06d}"
                output_video_dir.mkdir(parents=True, exist_ok=True)
                output_video_path = output_video_dir / f"{idx}-{prompt[:50].replace(' ', '_')}.mp4"
                write_video(output_video_path, video[0], fps=8)
                logger.info(f"Saved inference video to {output_video_path}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    def log_metrics(self, metrics):
        if self.is_main_process and self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.step)

    def log_iteration_time(self):
        if self.is_main_process:
            current_time = time.time()
            if self.previous_time is None:
                self.previous_time = current_time
            else:
                if self.writer:
                    self.writer.add_scalar(
                        "per_iteration_time", current_time - self.previous_time, self.step
                    )
                self.previous_time = current_time

    def maybe_run_gc(self):
        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logger.info("DistGarbageCollector: Running GC.")
            gc.collect()
            # gc only frees Python objects; without empty_cache the CUDA caching
            # allocator still holds the released blocks, so empty_cache pairs
            # naturally here to actually return them to the driver.
            torch.cuda.empty_cache()
