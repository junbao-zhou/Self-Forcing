from utils.wan_wrapper import WanVAEWrapper
from utils.distributed import launch_distributed_job
from datetime import timedelta
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.io as tvio
from tqdm import tqdm
import shutil
import torch
import time
import csv
import os

from utils.logging import (
    _current_node_rank,
    _current_process_rank,
    _configure_logging,
)
import logging

torch.set_grad_enabled(False)


class Config(BaseSettings):
    # cli_parse_args=True makes `Config()` read overrides straight from sys.argv,
    # so every field below is automatically a `--field-name` CLI flag with its
    # description as help text. No manual argparse needed.
    model_config = SettingsConfigDict(cli_parse_args=True)

    video_folder: Path = Field(
        default=Path("/publicdata/huggingface.co/datasets/nkp37/OpenVid-1M/video_folder"),
        description="Folder containing the source video files.",
    )
    csv_path: Path = Field(
        default=Path("/publicdata/huggingface.co/datasets/nkp37/OpenVid-1M/data/train/OpenVid-1M.csv"),
        description="CSV file listing video filename, caption, and frame count.",
    )
    output_folder: Path = Field(
        default=Path("/workspace/group_share/adc-perception-xbrain/zhoujb4/dataset/openvid_latent"),
        description="Folder where the encoded latent .pt files are written.",
    )
    height: int = Field(
        default=480,
        description="Target frame height before VAE encoding.",
    )
    width: int = Field(
        default=832,
        description="Target frame width before VAE encoding.",
    )
    latent_channels: int = Field(
        default=16,
        description="Number of channels in the VAE latent.",
    )
    video_id_list: Path | None = Field(
        default=None,
        description="Optional path to a text file of video ids (one per line, no .mp4 extension). If given, only these videos are processed.",
    )
    frame_downsample: int = Field(
        default=1,
        ge=1,
        description="Take 1 frame every N frames when loading video. 1 = no downsampling.",
    )
    save_id: str | None = Field(
        default=None,
        description="Identifier used in the log directory name.",
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Base directory for log files.",
    )


def downsampled_frame_count(total_frames: int, frame_downsample: int) -> int:
    # Taking 1 frame every `frame_downsample`, starting at index 0.
    return (total_frames + frame_downsample - 1) // frame_downsample


def expected_latent_shape(total_frames: int, config: Config) -> tuple[int, int, int, int, int]:
    # WAN VAE temporal compression: latent_F = 1 + (input_F - 1) // 4,
    # extra input frames beyond 4k+1 are dropped by the encoder.
    effective_frames = downsampled_frame_count(total_frames, config.frame_downsample)
    latent_frames = 1 + (effective_frames - 1) // 4
    return (1, latent_frames, config.latent_channels, config.height // 8, config.width // 8)


def existing_latent_is_valid(output_path: Path, expected_shape: tuple[int, ...]) -> bool:
    try:
        data = torch.load(output_path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as error:
        logging.info(f"Recomputing {output_path}: load failed ({error})")
        return False
    if not isinstance(data, dict) or "latent" not in data or "caption" not in data:
        logging.info(f"Recomputing {output_path}: bad keys")
        return False
    latent = data["latent"]
    if not torch.is_tensor(latent):
        logging.info(f"Recomputing {output_path}: latent is {type(latent)}")
        return False
    if tuple(latent.shape) != expected_shape:
        logging.info(f"Recomputing {output_path}: shape {tuple(latent.shape)} != expected {expected_shape}")
        return False
    return True


def load_and_resize_frames(video_path: Path, device: torch.device, config: Config) -> torch.Tensor:
    frames, _, _ = tvio.read_video(str(video_path), pts_unit="sec")
    if config.frame_downsample > 1:
        frames = frames[::config.frame_downsample]
    # [F, H, W, C] -> [F, C, H, W]
    frames = frames.to(device=device, dtype=torch.bfloat16).permute(0, 3, 1, 2)
    frames = F.interpolate(frames.float(), size=(config.height, config.width), mode="bilinear", align_corners=False)
    frames = frames.to(torch.bfloat16) / 255.0 * 2 - 1.0
    # [F, C, H, W] -> [1, C, F, H, W]
    return frames.permute(1, 0, 2, 3).unsqueeze(0)


def try_claim_video(claims_dir: Path, video_stem: str) -> bool:
    # O_CREAT|O_EXCL is atomic on POSIX (and on NFS for recent protocol versions),
    # so exactly one rank's create call succeeds for a given path.
    claim_path = claims_dir / f"{video_stem}.claim"
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def load_samples(wanted_ids: set[str] | None, config: Config) -> list[tuple[str, str, int]]:
    with config.csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (row["video"], row["caption"], int(row["frame"]))
            for row in reader
            if (wanted_ids is None or Path(row["video"]).stem in wanted_ids)
            and (config.video_folder / row["video"]).exists()
        ]


def process_one(
    sample: tuple[str, str, int],
    model: WanVAEWrapper,
    device: torch.device,
    config: Config,
) -> None:
    video_filename, caption, _ = sample
    output_path = config.output_folder / f"{Path(video_filename).stem}.pt"
    video_path = config.video_folder / video_filename

    try:
        t0 = time.time()
        video_tensor = load_and_resize_frames(video_path, device, config)
        t1 = time.time()
    except Exception as e:
        logging.info(f"Failed to read {video_path}: {e}")
        return

    latent = model.encode_to_latent(video_tensor)
    # latent.shape = [1, F, C, H, W]
    t2 = time.time()
    logging.info(
        f"load={t1-t0:.2f}s encode={t2-t1:.2f}s "
        f"{video_filename} {video_tensor.shape} -> {latent.shape}"
    )

    # Atomic save: write to a tmp path in the same directory, then os.replace.
    # POSIX rename within one filesystem is atomic, so an interrupted run leaves
    # either the old file or no file — never a half-written tensor.
    tmp_path = output_path.with_suffix(f".pt.tmp.{os.getpid()}")
    torch.save(
        {"latent": latent.cpu(), "caption": caption},
        tmp_path,
    )
    os.replace(tmp_path, output_path)


def main() -> None:
    config = Config()

    log_dir = config.log_dir / f"{config.save_id}-compute_vae_latent"
    log_dir.mkdir(exist_ok=True, parents=True)
    _configure_logging(
        log_dir / f"train-node{_current_node_rank()}-rank{_current_process_rank()}.log",
        logging_level=logging.DEBUG,
    )

    if "LOCAL_RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    launch_distributed_job()
    device = torch.cuda.current_device()

    # Separate gloo (CPU) group with a long timeout for the final barrier.
    # Pool mode reduces but does not eliminate end-of-job stragglers (the very
    # last claimed video on one rank can still be slow), so we keep the long
    # timeout to avoid NCCL's default 30-min kill.
    finish_barrier_group = dist.new_group(
        backend="gloo",
        timeout=timedelta(hours=24),
    )

    wanted_ids: set[str] | None = None
    if config.video_id_list is not None:
        wanted_ids = {
            line.strip()
            for line in config.video_id_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if dist.get_rank() == 0:
            logging.info(f"Filtering to {len(wanted_ids)} video ids from {config.video_id_list}")

    samples = load_samples(wanted_ids, config)

    config.output_folder.mkdir(parents=True, exist_ok=True)

    claims_dir = config.output_folder / ".claims"
    # Wipe claims dir + leftover *.pt.tmp.* files at startup so stale state
    # from a crashed previous run does not block videos from being retried.
    # This assumes no other instance of this script is running concurrently
    # against the same output_folder — concurrent runs would clobber each
    # other's claims and tmp files.
    if dist.get_rank() == 0:
        if claims_dir.exists():
            shutil.rmtree(claims_dir)
        claims_dir.mkdir(parents=True)
        orphan_tmp_files = list(config.output_folder.glob("*.pt.tmp.*"))
        for tmp_path in orphan_tmp_files:
            tmp_path.unlink()
        logging.info(
            f"Cleared claims dir: {claims_dir}; "
            f"removed {len(orphan_tmp_files)} orphan tmp files"
        )
    dist.barrier(group=finish_barrier_group)

    model = WanVAEWrapper(
        checkpoint_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ).to(device=device, dtype=torch.bfloat16)

    # Each rank scans all indices in order. Fast-skip already-valid outputs to
    # avoid touching the claims dir on a resume. Otherwise, try to atomically
    # claim the index; whichever rank wins the O_CREAT|O_EXCL race processes it,
    # the others move on. This decouples per-video latency from rank assignment,
    # so a few slow videos no longer block fast ranks.
    progress = tqdm(range(len(samples)), disable=dist.get_rank() != 0)
    for index in progress:
        sample = samples[index]
        _, _, total_frames = sample
        video_stem = Path(sample[0]).stem
        output_path = config.output_folder / f"{video_stem}.pt"
        expected_shape = expected_latent_shape(total_frames, config)

        t0 = time.time()
        is_skip_occupied_by_other = not try_claim_video(claims_dir, video_stem)
        t1 = time.time()
        # logging.info(f"claim {video_stem} {t1-t0:.6f}s occupied_by_other={is_skip_occupied_by_other}")
        if is_skip_occupied_by_other:
            continue
        # logging.info(f"claimed {video_stem} in {t1-t0:.6f}s")

        is_skip_valid_existing = output_path.exists() and existing_latent_is_valid(output_path, expected_shape)
        t2 = time.time()
        # logging.info(f"check existing {output_path} {t2-t1:.6f}s valid={is_skip_valid_existing}")
        if is_skip_valid_existing:
            continue

        logging.info(f"processing {video_stem}")

        process_one(
            sample=sample,
            model=model,
            device=device,
            config=config,
        )

    logging.info("finished its shard, waiting for others...")
    dist.barrier(group=finish_barrier_group)
    if dist.get_rank() == 0:
        logging.info("all ranks finished")


if __name__ == "__main__":
    main()
