import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle_with_sampler_epoch
from utils.distributed import (
    EMA_FSDP,
    fsdp_wrap,
    fsdp_state_dict,
)
from utils.misc import merge_dict_list
import torch.distributed as dist
from omegaconf import OmegaConf
from model import GAN
import torch
import os
from trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        config,
    ):

        super().__init__(config)

        # Configuration for discriminator warmup
        self.discriminator_warmup_steps = getattr(config, "discriminator_warmup_steps", 0)
        self.in_discriminator_warmup = self.step < self.discriminator_warmup_steps
        if self.in_discriminator_warmup and self.is_main_process:
            print(f"Starting with discriminator warmup for {self.discriminator_warmup_steps} steps")
        self.loss_scale = getattr(config, "loss_scale", 1.0)

        # Step 2: Initialize the model and optimizer
        self.model = GAN(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device,
                dtype=(torch.bfloat16 if config.mixed_precision else torch.float32),
            )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.gen_lr,
            betas=(config.beta1, config.beta2),
        )

        # Create separate parameter groups for the fake_score network
        # One group for parameters with "_cls_pred_branch" or "_gan_ca_blocks" in the name
        # and another group for all other parameters
        fake_score_params = []
        discriminator_params = []

        for name, param in self.model.fake_score.named_parameters():
            if param.requires_grad:
                if "_cls_pred_branch" in name or "_gan_ca_blocks" in name:
                    discriminator_params.append(param)
                else:
                    fake_score_params.append(param)

        # Use the special learning rate for the special parameter group
        # and the default critic learning rate for other parameters
        self.critic_param_groups = [
            {"params": fake_score_params, "lr": config.critic_lr},
            {
                "params": discriminator_params,
                "lr": config.critic_lr * config.discriminator_lr_multiplier,
            },
        ]
        if self.in_discriminator_warmup:
            self.critic_optimizer = torch.optim.AdamW(
                self.critic_param_groups, betas=(0.9, config.beta2_critic)
            )
        else:
            self.critic_optimizer = torch.optim.AdamW(
                self.critic_param_groups,
                betas=(config.beta1_critic, config.beta2_critic),
            )

        self.set_training_seed()

        # Step 3: Initialize the dataloader
        self.data_path = config.data_path
        dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))

        dataloader, sampler = self.build_distributed_dataloader(
            dataset,
            batch_size=config.batch_size,
        )

        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))

        self.dataloader = cycle_with_sampler_epoch(dataloader, sampler)

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
        # Initialize generator from pretrained weights if provided.
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=True)
        if hasattr(config, "load"):
            resume_ckpt_path_critic = os.path.join(config.load, "critic")
            resume_ckpt_path_generator = os.path.join(config.load, "generator")
        else:
            resume_ckpt_path_critic = "none"
            resume_ckpt_path_generator = "none"

        # Resume critic/generator training state if checkpointer paths are configured.
        _, _ = self.checkpointer_critic.try_best_load(
            resume_ckpt_path=resume_ckpt_path_critic,
        )
        self.step, _ = self.checkpointer_generator.try_best_load(
            resume_ckpt_path=resume_ckpt_path_generator,
            force_start_w_ema=config.force_start_w_ema,
            force_reset_zero_step=config.force_reset_zero_step,
            force_reinit_ema=config.force_reinit_ema,
            skip_optimizer_scheduler=config.skip_optimizer_scheduler,
        )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)

    def save(
        self,
    ):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        self.save_checkpoint(state_dict)

    def fwdbwd_one_step(
        self,
        batch,
        train_generator,
    ):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]  # next(self.dataloader)
        if "ode_latent" in batch:
            clean_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
        else:
            frames = batch["frames"].to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(frames).to(
                    device=self.device, dtype=self.dtype
                )

            image_latent = clean_latent[
                :,
                0:1,
            ]

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

        mini_bs, full_bs = (
            batch["mini_bs"],
            batch["full_bs"],
        )

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            gan_G_loss = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
            )

            loss_ratio = mini_bs * self.world_size / full_bs
            total_loss = gan_G_loss * loss_ratio * self.loss_scale

            total_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)

            generator_log_dict = {
                "generator_grad_norm": generator_grad_norm,
                "gan_G_loss": gan_G_loss,
            }

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        (gan_D_loss, r1_loss, r2_loss), critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            real_image_or_video=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
        )

        loss_ratio = mini_bs * dist.get_world_size() / full_bs
        total_loss = (gan_D_loss + 0.5 * (r1_loss + r2_loss)) * loss_ratio * self.loss_scale

        total_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)

        critic_log_dict.update(
            {
                "critic_grad_norm": critic_grad_norm,
                "gan_D_loss": gan_D_loss,
                "r1_loss": r1_loss,
                "r2_loss": r2_loss,
            }
        )

        return critic_log_dict

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
        start_step = self.step

        while self.step < self.config.total_training_steps:
            logger.info(f"{self.step = } , starting training step...")
            # Run inference
            if (
                self.config.inference_interval > 0
                and self.step % self.config.inference_interval == 0
            ):
                self.run_inference(self.model.generator)

            if (
                self.step == self.discriminator_warmup_steps
                and self.discriminator_warmup_steps != 0
            ):
                print("Resetting critic optimizer")
                del self.critic_optimizer
                gc.collect()
                torch.cuda.empty_cache()
                # Create new optimizers
                self.critic_optimizer = torch.optim.AdamW(
                    self.critic_param_groups,
                    betas=(self.config.beta1_critic, self.config.beta2_critic),
                )
                # Update checkpointer references
                self.checkpointer_critic.optimizer = self.critic_optimizer
            # Check if we're in the discriminator warmup phase
            self.in_discriminator_warmup = self.step < self.discriminator_warmup_steps

            # Only update generator and critic outside the warmup phase
            TRAIN_GENERATOR = (
                not self.in_discriminator_warmup
                and self.step % self.config.dfake_gen_update_ratio == 0
            )

            # Train the generator (only outside warmup phase)
            if TRAIN_GENERATOR:
                self.model.fake_score.requires_grad_(False)
                self.model.generator.requires_grad_(True)
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for ii, mini_batch in enumerate(self.dataloader.next()):
                    extra = self.fwdbwd_one_step(mini_batch, True)
                    extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
            else:
                generator_log_dict = {}

            # Train the critic/discriminator
            if self.in_discriminator_warmup:
                # During warmup, only allow gradient for discriminator params
                self.model.generator.requires_grad_(False)
                self.model.fake_score.requires_grad_(False)

                # Enable gradient only for discriminator params
                for name, param in self.model.fake_score.named_parameters():
                    if "_cls_pred_branch" in name or "_gan_ca_blocks" in name:
                        param.requires_grad_(True)
            else:
                # Normal training mode
                self.model.generator.requires_grad_(False)
                self.model.fake_score.requires_grad_(True)

            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # If we just finished warmup, print a message
            if self.is_main_process and self.step == self.discriminator_warmup_steps:
                print(
                    f"Finished discriminator warmup after {self.discriminator_warmup_steps} steps"
                )

            # Create EMA params (if not already created)
            if (
                (self.step >= self.config.ema_start_step)
                and (self.generator_ema is None)
                and (self.config.ema_weight > 0)
            ):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (
                (not self.config.no_save)
                and (self.step - start_step) > 0
                and self.step % self.config.log_iters == 0
            ):
                gc.collect()
                torch.cuda.empty_cache()
                self.save()
                gc.collect()
                torch.cuda.empty_cache()

            # Logging
            logging_loss_dict = {
                "generator_grad_norm": generator_log_dict["generator_grad_norm"],
                "critic_grad_norm": critic_log_dict["critic_grad_norm"],
                "real_logit": critic_log_dict["noisy_real_logit"],
                "fake_logit": critic_log_dict["noisy_fake_logit"],
                "r1_loss": critic_log_dict["r1_loss"],
                "r2_loss": critic_log_dict["r2_loss"],
            }
            if TRAIN_GENERATOR:
                logging_loss_dict.update(
                    {
                        "generator_grad_norm": generator_log_dict["generator_grad_norm"],
                    }
                )
            self.all_gather_dict(logging_loss_dict)
            logging_loss_dict["diff_logit"] = (
                logging_loss_dict["real_logit"] - logging_loss_dict["fake_logit"]
            )
            logging_loss_dict["reg_loss"] = 0.5 * (
                logging_loss_dict["r1_loss"] + logging_loss_dict["r2_loss"]
            )

            if self.is_main_process:
                if self.in_discriminator_warmup:
                    warmup_status = f"[WARMUP {self.step}/{self.discriminator_warmup_steps}] Training only discriminator params"
                    print(warmup_status)
                    logging_loss_dict.update({"warmup_status": 1.0})

                self.log_metrics(logging_loss_dict)

            self.maybe_run_gc()
            torch.cuda.empty_cache()
            self.log_iteration_time()

    def all_gather_dict(
        self,
        target_dict,
    ):
        for key, value in target_dict.items():
            gathered_value = torch.zeros(
                [self.world_size, *value.shape],
                dtype=value.dtype,
                device=self.device,
            )
            dist.all_gather_into_tensor(gathered_value, value)
            avg_value = gathered_value.mean().item()
            target_dict[key] = avg_value
