import gc
import logging
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.distributed import fsdp_state_dict
from utils.misc import set_seed


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.device = torch.cuda.current_device()
        self.is_main_process = dist.get_rank() == 0
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + dist.get_rank())

        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=os.path.join(config.logdir, "tensorboard"))
        else:
            self.writer = None

        self.output_path = config.logdir
        self.max_grad_norm = 10.0
        self.previous_time = None

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
            logging.info(
                f"Model saved to {os.path.join(self.output_path, f'checkpoint_model_{self.step:06d}', 'model.pt')}"
            )

            max_checkpoints = self.config.max_checkpoints
            checkpoints = sorted(
                [d for d in os.listdir(self.output_path) if d.startswith("checkpoint_model_")]
            )
            if len(checkpoints) > max_checkpoints:
                for old_ckpt in checkpoints[:-max_checkpoints]:
                    shutil.rmtree(os.path.join(self.output_path, old_ckpt))
                    logging.info(f"Deleted old checkpoint: {old_ckpt}")

    def run_inference(self, generator):
        logging.info("Gathering generator state dict for inference...")
        generator_state = fsdp_state_dict(generator)

        if not self.is_main_process:
            return

        with open("prompts/MovieGenVideoBench_extended.txt", "r") as f:
            prompts = [line.strip() for line in f.readlines()[0:7:2]]

        pipeline = CausalInferencePipeline(self.config, device=self.device)
        pipeline.generator.load_state_dict(generator_state)
        pipeline = pipeline.to(dtype=torch.bfloat16)

        with torch.no_grad():
            for idx, prompt in enumerate(prompts):
                noise = torch.randn([1, 21, 16, 60, 104], device=self.device, dtype=torch.bfloat16)
                video, _ = pipeline.inference(
                    noise=noise, text_prompts=[prompt], return_latents=True
                )
                video = rearrange(video, "b t c h w -> b t h w c").cpu()
                video = 255.0 * video
                output_video_dir = Path(self.output_path) / f"inference_videos_step{self.step:06d}"
                output_video_dir.mkdir(parents=True, exist_ok=True)
                output_video_path = output_video_dir / f"{idx}-{prompt[:50].replace(' ', '_')}.mp4"
                write_video(output_video_path, video[0], fps=8)
                logging.info(f"Saved inference video to {output_video_path}")

        del pipeline
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
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()
