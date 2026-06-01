"""Scan all .pt latent files; print any that are corrupted or have unexpected schema/shape.

With --mode random, instead picks one random .pt latent, decodes it via the VAE, and
saves the decoded video next to the source GT video plus the caption text, for visual
verification of latent correctness.
"""

import csv
import multiprocessing
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import torch
import torchvision.io as tvio
from tqdm import tqdm
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # cli_parse_args=True makes `Config()` read overrides straight from sys.argv,
    # so every field below is automatically a `--field-name` CLI flag.
    # cli_implicit_flags=True lets booleans act as bare flags (e.g. --check_numeric).
    model_config = SettingsConfigDict(cli_parse_args=True, cli_implicit_flags=True)

    height: int = Field(
        default=480,
        description="Expected frame height the latents were encoded at.",
    )
    width: int = Field(
        default=832,
        description="Expected frame width the latents were encoded at.",
    )
    latent_channels: int = Field(
        default=16,
        description="Expected number of channels in the VAE latent.",
    )
    frame_downsample: int = Field(
        default=1,
        ge=1,
        description="Frame downsample factor the latents were encoded with. 1 = no downsampling.",
    )
    video_folder: Path = Field(
        default=Path("/publicdata/huggingface.co/datasets/nkp37/OpenVid-1M/video_folder"),
        description="Folder containing the source video files.",
    )
    csv_path: Path = Field(
        default=Path("/publicdata/huggingface.co/datasets/nkp37/OpenVid-1M/data/train/OpenVid-1M.csv"),
        description="CSV file listing video filename, caption, and frame count.",
    )
    latent_folder: Path = Field(
        default=Path("/workspace/group_share/adc-perception-xbrain/zhoujb4/dataset/openvid_latent"),
        description="Folder containing the encoded latent .pt files to check.",
    )
    mode: Literal["scan", "random"] = Field(
        default="scan",
        description="scan = check every latent; random = decode one random latent for visual verification.",
    )
    num_workers: int = Field(
        default=192,
        description="Number of worker processes used by the scan.",
    )
    bad_latent_record_file: Path = Field(
        default=Path("bad_latents.txt"),
        description="File where paths of corrupted/mismatched latents are recorded.",
    )
    verify_folder: Path = Field(
        default=Path("outputs/check_latents_random"),
        description="Output folder for the decoded video / caption in random mode.",
    )
    device: str = Field(
        default="cuda",
        description="Device used to decode the latent in random mode.",
    )
    check_numeric: bool = Field(
        default=False,
        description="Also check each latent for NaN/Inf. Significantly slower (forces a full tensor scan).",
    )


# Per-worker GPU assignment, set by `_init_worker` in each Pool worker.
# None means "do numeric checks on CPU" (no GPUs available, or check disabled).
_WORKER_GPU_ID: int | None = None


def _init_worker(gpu_ids: list[int]) -> None:
    global _WORKER_GPU_ID
    if not gpu_ids:
        return
    # multiprocessing.Pool gives each worker a 1-indexed identity tuple.
    # Fall back to PID hashing if for some reason _identity is empty.
    identity = multiprocessing.current_process()._identity
    print(f"Worker identity: {identity = }")
    worker_index = identity[0] - 1 if identity else os.getpid()
    print(f"Assigning worker {worker_index = }")
    _WORKER_GPU_ID = gpu_ids[worker_index % len(gpu_ids)]
    print(f"Worker {worker_index = } assigned GPU {_WORKER_GPU_ID = }")


def expected_latent_shape(total_frames: int, config: Config) -> tuple[int, int, int, int, int]:
    # Mirror compute_vae_latent_pool: WAN VAE temporal compression yields
    # latent_F = 1 + (effective_F - 1) // 4, where effective_F is the
    # downsampled input frame count.
    effective_frames = (total_frames + config.frame_downsample - 1) // config.frame_downsample
    latent_frames = 1 + (effective_frames - 1) // 4
    return (1, latent_frames, config.latent_channels, config.height // 8, config.width // 8)


def inspect(args: tuple[str, tuple[int, ...], bool]) -> tuple[str, str | None, str, tuple]:
    path, expected_shape, check_numeric = args
    try:
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as error:
        return path, f"load failed: {error}", "", ()

    if not isinstance(data, dict) or "latent" not in data or "caption" not in data:
        return (
            path,
            f"bad keys: {list(data.keys()) if isinstance(data, dict) else type(data)}",
            "",
            (),
        )

    caption = data["caption"] if isinstance(data["caption"], str) else ""
    latent = data["latent"]
    shape = tuple(latent.shape) if torch.is_tensor(latent) else ()

    if not isinstance(data["caption"], str) or not data["caption"]:
        return path, f"bad caption: {data['caption']!r}", caption, shape

    if not torch.is_tensor(latent):
        return path, f"latent is {type(latent)}", caption, shape

    if shape != expected_shape:
        return path, f"shape {shape} != expected {expected_shape}", caption, shape

    if latent.dtype is not torch.float32:
        return path, f"unexpected dtype {latent.dtype}", caption, shape

    if check_numeric:
        device = torch.device(f"cuda:{_WORKER_GPU_ID}") if _WORKER_GPU_ID is not None else torch.device("cpu")
        latent_on_device = latent.to(device, non_blocking=False)
        if torch.isnan(latent_on_device).any() or torch.isinf(latent_on_device).any():
            return path, "NaN/Inf in latent", caption, shape

    return path, None, caption, shape


def save_video(path: Path, frames: torch.Tensor, fps: int = 16) -> None:
    # frames: [T, C, H, W] in [-1, 1] or [0, 1]; write_video wants [T, H, W, C] uint8.
    frames = frames.detach().float().clamp(-1.0, 1.0)
    frames = ((frames + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).contiguous().cpu()
    tvio.write_video(str(path), frames, fps=fps)


def random_check(config: Config, device: torch.device) -> None:
    from utils.wan_wrapper import WanVAEWrapper

    output_dir = config.verify_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [entry.name for entry in os.scandir(config.latent_folder) if entry.name.endswith(".pt")]
    latent_name = random.choice(candidates)
    latent_path = config.latent_folder / latent_name
    video_id = latent_path.stem
    print(f"Picked latent: {latent_path}")

    data = torch.load(latent_path, map_location="cpu", mmap=True, weights_only=False)
    latent = data["latent"].to(device=device, dtype=torch.bfloat16)
    caption = data["caption"]

    video_path = config.video_folder / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"GT video not found: {video_path}")

    model = WanVAEWrapper(
        checkpoint_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ).to(device=device, dtype=torch.bfloat16)
    decoded = model.decode_to_pixel(latent)  # [1, T, C, H, W]
    decoded_frames = decoded.squeeze(0)

    gt_out = output_dir / f"{video_id}_gt.mp4"
    decoded_out = output_dir / f"{video_id}_decoded.mp4"
    caption_out = output_dir / f"{video_id}_caption.txt"

    shutil.copy(video_path, gt_out)
    save_video(decoded_out, decoded_frames, fps=16)
    caption_out.write_text(caption, encoding="utf-8")

    print(f"GT video:    {gt_out}")
    print(f"Decoded:     {decoded_out}")
    print(f"Caption:     {caption_out}")
    print(f"Caption text: {caption}")


def get_video_frames_dict(config: Config) -> dict[str, int]:
    with config.csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {Path(row["video"]).stem: int(row["frame"]) for row in reader}


def scan(config: Config) -> None:
    video_frames_dict = get_video_frames_dict(config)

    paths = sorted(
        str(config.latent_folder / entry.name)
        for entry in os.scandir(config.latent_folder)
        if entry.name.endswith(".pt")
    )

    latent_inspect_args: list[tuple[str, tuple[int, ...], bool]] = []
    for path in paths:
        stem = Path(path).stem
        if stem not in video_frames_dict:
            tqdm.write(f"BAD {path}\tno CSV entry for stem {stem!r}")
            continue
        latent_inspect_args.append(
            (path, expected_latent_shape(video_frames_dict[stem], config), config.check_numeric)
        )

    # Only enumerate GPUs when we will actually use them; avoids initializing
    # CUDA in the workers for a shape-only scan.
    gpu_ids = list(range(torch.cuda.device_count())) if config.check_numeric else []
    if config.check_numeric:
        tqdm.write(f"check_numeric=True, dispatching workers across GPUs: {gpu_ids}")

    csv_path = config.latent_folder / "latents.csv"
    tqdm.write(
        f"Scanning {len(latent_inspect_args)} files -> bad list: {config.bad_latent_record_file}, csv: {csv_path}"
    )

    with (
        open(config.bad_latent_record_file, "w") as bad_record_file,
        open(csv_path, "w", newline="") as csv_file,
        Pool(processes=config.num_workers, initializer=_init_worker, initargs=(gpu_ids,)) as pool,
    ):
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["filename", "caption", "shape"])
        results = pool.imap_unordered(inspect, latent_inspect_args, chunksize=1)
        for path, reason, caption, shape in tqdm(results, total=len(latent_inspect_args), desc="scan"):
            csv_writer.writerow([os.path.basename(path), caption, str(shape)])
            if reason is not None:
                tqdm.write(f"BAD {path}\t{reason}")
                bad_record_file.write(f"{path}\t{reason}\n")
                bad_record_file.flush()


def main() -> None:
    config = Config()

    if config.mode == "random":
        random_check(config, torch.device(config.device))
    else:
        scan(config)


if __name__ == "__main__":
    main()
