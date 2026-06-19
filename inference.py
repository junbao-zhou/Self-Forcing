import os
import time

from utils.logging import (
    _configure_logging,
    log_environment_versions,
    string_to_logging_level,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")

class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    config_dir: str = "configs"
    config_name: str = "self_forcing_dmd_vsink"
    checkpoint_path: str = "./checkpoints/self_forcing_dmd.pt"
    data_path: str = "prompts/MovieGenVideoBench_extended.txt"
    extended_prompt_path: str | None = None
    output_folder: str = f"outputs/inference-{timestamp_str}/21-self_forcing_dmd_vsink-seed42"
    num_output_frames: int = 21
    i2v: bool = False
    use_ema: bool = True
    seed: int = 42
    num_samples: int = 1
    max_video_num: int = 20


args = InferenceConfig()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize(version_base=None, config_path=args.config_dir):
    config = compose(config_name=args.config_name)
print(f"{config = }")

logdir = Path(args.output_folder)
logdir.mkdir(parents=True, exist_ok=True)

_configure_logging(
    logdir / f"inference.log",
    logging_level=string_to_logging_level(config.logging_level),
)
log_environment_versions()

# Initialize pipeline
if hasattr(config, "denoising_step_list"):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(
        state_dict["generator" if not args.use_ema else "generator_ema"]
    )

pipeline = pipeline.to(dtype=torch.bfloat16)

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose(
        [
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(
        prompt_path=args.data_path,
        extended_prompt_path=args.extended_prompt_path,
    )
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    if args.max_video_num != -1 and idx >= args.max_video_num:
        break

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    set_seed(args.seed)
    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch["prompts"][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = (
            batch["image"]
            .squeeze(0)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(device=device, dtype=torch.bfloat16)
        )

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(
            device=device, dtype=torch.bfloat16
        )
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch["prompts"][0]
        extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )

    set_seed(args.seed)
    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            output_path = os.path.join(
                args.output_folder,
                f"{idx}-{prompt[:50].replace(' ', '_')}-{seed_idx}_{model}.mp4",
            )
            write_video(output_path, video[seed_idx], fps=16)
