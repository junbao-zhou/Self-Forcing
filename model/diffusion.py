import logging
from typing import Tuple
import torch
import torch.nn.functional as F

from model.base import SelfForcingModel
from pipeline import SelfForcingTrainingPipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusion(SelfForcingModel):
    """
    Self-forcing diffusion training without distillation.

    Like DMD, the student is rolled out auto-regressively via `_run_generator`
    (KV cache + exit_flag), so train and test see the same input distribution.
    Unlike DMD, there is no real_score / fake_score / KL gradient — the x0
    estimates produced by the rollout are supervised directly with MSE against
    the dataset's ground-truth clean latent.
    """

    def __init__(
        self,
        args,
        device,
    ):
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Lazily built once FSDP-wrapped modules are passed in by the trainer.
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        self.num_train_timestep = args.num_train_timestep
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

    def _initialize_models(self, args, device):
        # Override SelfForcingModel/BaseModel to skip real_score and fake_score:
        # this trainer has no distillation, so those modules would just waste memory.
        self.generator = WanDiffusionWrapper(
            config_path=args.generator_config_path,
            **getattr(args, "model_kwargs", {}),
            is_causal=True,
        )
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder(checkpoint_path=args.text_encoder_checkpoint_path)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(checkpoint_path=args.vae_checkpoint_path)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Self-forcing rollout + MSE supervision against the ground-truth clean latent.

        Input:
            - image_or_video_shape: [B, F, C, H, W].
            - conditional_dict: text-encoder outputs (and optional initial_latent for i2v).
            - unconditional_dict: kept for API compatibility; unused here.
            - clean_latent: dataset ground-truth latent, shape [B, F, C, H, W].
            - initial_latent: i2v image latent; ignored when args.i2v is False.
        """
        logging.debug(
            f"""
    {image_or_video_shape = },
    {conditional_dict.keys() = },
    {clean_latent.shape = },
    initial_latent={None if initial_latent is None else initial_latent.shape},
"""
        )

        # Step 1: self-forcing auto-regressive rollout. pred_image stacks per-block
        # x0 estimates taken at the (rank-synced) exit_flag denoising step.
        pred_image, gradient_mask, _, _ = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent if self.args.i2v else None,
        )
        logging.debug(
            f"_run_generator -> {pred_image.shape = }, {gradient_mask is None = }"
        )

        # Step 2: direct MSE in x0 space against the dataset latent.
        pred_image_f = pred_image.float()
        clean_latent_f = clean_latent.float()
        if gradient_mask is not None:
            loss = F.mse_loss(pred_image_f[gradient_mask], clean_latent_f[gradient_mask])
        else:
            loss = F.mse_loss(pred_image_f, clean_latent_f)

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": pred_image.detach(),
        }
        return loss, log_dict
