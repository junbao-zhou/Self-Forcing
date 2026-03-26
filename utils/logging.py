

import logging
from pathlib import Path
import time
import torch.distributed as dist
import os


def _current_node_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() // dist.get_world_size()
    # fallback to env vars set by torchrun/launch
    return int(os.environ.get("NODE_RANK", 0))


def _current_process_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    # fallback to env vars set by torchrun/launch
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def _add_rank_to_record():
    factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.rank = _current_process_rank()
        return record

    logging.setLogRecordFactory(record_factory)


def _configure_logging(
    logfile: Path | str,
):
    """
    Hydra installs its own logging handlers before main() runs.
    This function forcefully replaces them so our file + format take effect.
    """
    _add_rank_to_record()

    fmt = logging.Formatter(
        "[rank:{rank}] [{levelname}] [{asctime}] : {message}",
        style="{",
    )

    file_handler = logging.FileHandler(logfile, mode="a")
    stream_handler = logging.StreamHandler()

    for handler in [file_handler, stream_handler]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(fmt)

    # Replace existing handlers (Hydra already configured them)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [file_handler, stream_handler]