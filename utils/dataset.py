from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import csv
import cv2
import logging


class TextDataset(Dataset):
    def __init__(
        self,
        prompt_path,
        extended_prompt_path=None,
    ):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(
        self,
    ):
        return len(self.prompt_list)

    def __getitem__(
        self,
        idx,
    ):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
    ):
        self.env = lmdb.open(data_path, readonly=True, lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, "latents")
        self.max_pair = max_pair

    def __len__(
        self,
    ):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(
        self,
        idx,
    ):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env, "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
        }


class ShardingLMDBDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
    ):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, "latents")
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(
        self,
    ):
        return len(self.index)

    def __getitem__(
        self,
        idx,
    ):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents",
            np.float16,
            local_idx,
            shape=self.latents_shape[shard_id][1:],
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(self.envs[shard_id], "prompts", str, local_idx)

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None,
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob("target_crop_info_*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split("_")[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(
        self,
    ):
        return len(self.metadata)

    def __getitem__(
        self,
        idx,
    ):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item["file_name"]
        image = Image.open(image_path).convert("RGB")

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "prompts": item["caption"],
            "target_bbox": item["target_crop"]["target_bbox"],
            "target_ratio": item["target_crop"]["target_ratio"],
            "type": item["type"],
            "origin_size": (item["origin_width"], item["origin_height"]),
            "idx": idx,
        }


class OpenVidDataset(Dataset):
    def __init__(
        self,
        video_folder: str,
        csv_path: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
    ):
        logging.debug(
            f"""
    {video_folder = },
    {csv_path = },
    {num_frames = },
    {height = },
    {width = },
"""
        )
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.height = height
        self.width = width

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.samples = [
                (row["video"], row["caption"], int(row["frame"]))
                for row in reader
                if int(row["frame"]) >= num_frames
                and os.path.exists(os.path.join(video_folder, row["video"]))
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        video_filename, caption, total_frames = self.samples[idx]
        video_path = os.path.join(self.video_folder, video_filename)

        cap = cv2.VideoCapture(video_path)
        start = np.random.randint(0, total_frames - self.num_frames + 1)
        indices = range(start, start + self.num_frames)

        frames = []
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.width, self.height))
            frames.append(frame)
        cap.release()

        # [F, H, W, C] -> [C, F, H, W], normalized to [-1, 1]
        frames_array = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
        frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2)

        return {
            "frames": frames_tensor,  # [C, F, H, W]
            "prompts": caption,
        }


class OpenVidLatentDataset(Dataset):
    """
    Dataset of OpenVid VAE latents pre-computed by compute_vae_latent.py.

    Each .pt file stores a dict {"latent": Tensor[1, F, C, H, W], "caption": str}.
    Returned `ode_latent` keeps the leading singleton step dim so it is a
    drop-in replacement for `ShardingLMDBDataset` in the trainer
    (which indexes `ode_latent[:, -1]` to get the clean latent).
    """

    def __init__(
        self,
        latent_folder: str,
        max_pair: int = int(1e8),
    ):
        logging.debug(
            f"""
    {latent_folder = },
    {max_pair = },
"""
        )
        self.latent_folder = latent_folder
        self.latent_files = sorted(
            entry.name
            for entry in os.scandir(latent_folder)
            if entry.name.endswith(".pt")
        )
        self.latent_files = self.latent_files[:max_pair]

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.latent_folder, self.latent_files[idx])
        data = torch.load(path, map_location="cpu", weights_only=False)
        latent = data["latent"]
        if not torch.is_tensor(latent):
            latent = torch.tensor(latent)
        # File stores [1, F, C, H, W]; treat the leading 1 as the denoising-step dim
        # so downstream code that does `ode_latent[:, -1]` still works.
        return {
            "prompts": data["caption"],
            "ode_latent": latent.to(torch.float32),
        }


def cycle(
    dl,
):
    while True:
        for data in dl:
            yield data
