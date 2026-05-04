import gc
import logging

from model import CausalDiffusion
from utils.dataset import OpenVidDataset, OpenVidLatentDataset, cycle
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import os
from trainer.base import BaseTrainer

from utils.distributed import (
    EMA_FSDP,
    barrier,
    fsdp_wrap,
    fsdp_state_dict,
    launch_distributed_job,
)


class Trainer(BaseTrainer):
    def __init__(
        self,
        config,
    ):
        logging.debug(
            f"""
    {config = }
"""
        )
        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        super().__init__(config)
        self.causal = config.causal

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device,
                dtype=(torch.bfloat16 if config.mixed_precision else torch.float32),
            )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        # Step 3: Initialize the dataloader
        if config.load_raw_video:
            latent_frames = config.image_or_video_shape[1]
            video_frames = (latent_frames - 1) * 4 + 1
            dataset = OpenVidDataset(
                video_folder=config.video_folder,
                csv_path=config.csv_path,
                num_frames=video_frames,
            )
        else:
            dataset = OpenVidLatentDataset(config.latent_folder)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8,
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

    def save(
        self,
    ):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        self.save_checkpoint(state_dict)

    def train_one_step(
        self,
        batch,
    ):
        logging.debug(
            f"""
    {self.step = }
    {batch.keys() = }
"""
        )
        self.log_iters = 1

        if self.step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(device=self.device, dtype=self.dtype)
            logging.debug(f"{frames.shape = }")
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(frames).to(
                    device=self.device, dtype=self.dtype
                )

        logging.debug(f"{clean_latent.shape = }")
        image_latent = clean_latent[
            :,
            0:1,
        ]
        logging.debug(f"{image_latent.shape = }")

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        logging.debug(f"Start generator_loss computation")
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
        )
        self.generator_optimizer.zero_grad()
        # Silence DEBUG/INFO logs during backward: gradient checkpointing reruns
        # each transformer block under autograd, which bypasses the outer
        # for-loop's logging.disable wrapping in CausalWanModel and floods the log.
        # Disable up to INFO only (not CRITICAL) so WARNING/ERROR/CRITICAL still
        # get through — important for `logging.exception` if backward throws OOM.
        prev_disable = logging.root.manager.disable
        logging.disable(logging.INFO)
        generator_loss.backward()
        logging.disable(prev_disable)
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()
        if self.generator_ema is not None:
            self.generator_ema.update(self.model.generator)

        # Increment the step since we finished gradient update
        self.step += 1

        # Create EMA params (if not already created)
        if (
            (self.step >= self.config.ema_start_step)
            and (self.generator_ema is None)
            and (self.config.ema_weight > 0)
        ):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        self.log_metrics(wandb_loss_dict)
        self.maybe_run_gc()

    def generate_video(
        self,
        pipeline,
        prompts,
        image=None,
    ):
        batch_size = len(prompts)
        sampled_noise = torch.randn([batch_size, 21, 16, 60, 104], device="cuda", dtype=self.dtype)
        video, _ = pipeline.inference(
            noise=sampled_noise, text_prompts=prompts, return_latents=True
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(
        self,
    ):
        while True:
            # Run inference
            if (
                self.config.inference_interval > 0
                and self.step % self.config.inference_interval == 0
            ):
                gc.collect()
                torch.cuda.empty_cache()
                self.run_inference(self.model.generator)
                gc.collect()
                torch.cuda.empty_cache()

            batch = next(self.dataloader)
            self.train_one_step(batch)
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                gc.collect()
                torch.cuda.empty_cache()
                self.save()
                gc.collect()
                torch.cuda.empty_cache()

            barrier()
            self.log_iteration_time()
