from utils.logging import logger

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block=3,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        **kwargs
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[
                :-1
            ]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(
        self,
        num_blocks,
        num_denoising_steps,
        device,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device,
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.zeros(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        result = indices.tolist()
        return result

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict
    ) -> torch.Tensor:
        logger.debug(f"""
    {noise.shape = },
    initial_latent = {None if initial_latent is None else initial_latent.shape}
""")

        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        logger.debug(f"Computed {num_blocks=}, {num_frames=}, {self.num_frame_per_block=}")

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        # Step 1: Initialize KV cache lazily on first call; reuse + reset afterwards.
        # Re-allocating ~12 GB (per-rank, 30 layers × 2 × bf16) of KV cache every
        # step caused allocator churn / fragmentation OOM in 2-node 16-GPU runs.
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )
        else:
            self._reset_kv_cache()
            self._reset_crossattn_cache()

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device
        )

        logger.debug(f"After generate_and_sync_list: {exit_flags = }, {len(exit_flags) = }")

        if len(exit_flags) == 0:
            raise RuntimeError(f"exit_flags is empty! {num_blocks = }, {all_num_frames = }")

        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        logger.debug(f"Starting generating blocks, {num_blocks=}, {len(all_num_frames)=}, {exit_flags=}")
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :,
                current_start_frame
                - num_input_frames : current_start_frame
                + current_num_frames
                - num_input_frames,
            ]
            logger.debug(f"{block_index = }, {current_num_frames = }, {noisy_input.shape = }, {current_start_frame = }, {num_input_frames = }")

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                logger.debug(f"timestep index = {index}, {current_timestep = }")
                logger.debug(f"{self.same_step_across_blocks = }")
                if self.same_step_across_blocks:
                    exit_flag = index == exit_flags[0]
                else:
                    exit_flag = (
                        index == exit_flags[block_index]
                    )  # Only backprop at the randomly selected timestep (consistent across all ranks)
                logger.debug(f"{exit_flag = }")
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if not exit_flag:
                    logger.debug(f"No exit at this step, running with torch.no_grad()")
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    logger.debug(f"Exit at this step")
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        logger.debug(f"{current_start_frame = } < {start_gradient_frame_index = }, running with torch.no_grad()")
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        logger.debug(f"{current_start_frame = } >= {start_gradient_frame_index = }, running with gradients enabled")
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                    break

            # Step 3.2: record the model's output
            output[
                :,
                current_start_frame : current_start_frame + current_num_frames,
            ] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            logger.debug(f"{self.context_noise = }")
            context_timestep = (
                torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * self.context_noise
            )
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                self.context_noise
                * torch.ones(
                    [batch_size * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_pred.shape[:2])
            logger.debug(f"Rerunning the model with {context_timestep[0,0].item() = } to update the cache")
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        logger.debug(f"Computing denoised_timestep")

        if not self.same_step_across_blocks:
            logger.debug(f"{self.same_step_across_blocks = }")
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            logger.debug(f"{exit_flags[0] = } == {len(self.denoising_step_list) - 1 = }")
            denoised_timestep_to = 0
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0]].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )
        else:
            denoised_timestep_to = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0] + 1].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0]].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )

        logger.debug(f"Computed {denoised_timestep_from = }, {denoised_timestep_to = }")

        if return_sim_step:
            return (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                exit_flags[0] + 1,
            )

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, self.kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, self.kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(
        self,
        batch_size,
        dtype,
        device,
    ):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def _reset_kv_cache(self):
        """
        Reset KV cache between training steps without re-allocating the K/V buffers.
        Re-uses the already-allocated tensors, zeros the buffers and the index
        counters.

        Note: the model writes K/V into the cache via in-place index assignment
        (`kv_cache["k"][:, s:e] = roped_key`). When `roped_key` carries grad
        (the gradient block under self-forcing), this in-place op attaches a
        `grad_fn` (IndexPut) to the cache tensor and turns it from a leaf into
        a non-leaf with `requires_grad=True`. Backward frees graph memory but
        does not clear `grad_fn`, so `is_leaf` stays False into the next step
        and `requires_grad_(False)` would error.
        Calling `detach_()` first severs the lingering autograd link and makes
        the tensor a leaf again, which is why this is the original author's
        reason for not enabling lazy KV-cache reset.
        """
        logger.debug(f"{type(self).__name__}._reset_kv_cache")
        for block_index in range(len(self.kv_cache1)):
            cache = self.kv_cache1[block_index]
            for key in ("k", "v"):
                tensor = cache[key]
                tensor.detach_()
                if tensor.requires_grad:
                    tensor.requires_grad_(False)
                if tensor.grad is not None:
                    tensor.grad = None
                tensor.zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()

    def _reset_crossattn_cache(self):
        """
        Reset cross-attn cache between training steps. Same in-place strategy
        as `_reset_kv_cache`: detach lingering autograd links first, then keep
        and zero the buffers, flip is_init, scrub grads.
        """
        logger.debug(f"{type(self).__name__}._reset_crossattn_cache")
        for block_index in range(self.num_transformer_blocks):
            cache = self.crossattn_cache[block_index]
            for key in ("k", "v"):
                tensor = cache[key]
                tensor.detach_()
                if tensor.requires_grad:
                    tensor.requires_grad_(False)
                if tensor.grad is not None:
                    tensor.grad = None
                tensor.zero_()
            cache["is_init"] = False
