

import logging
from pathlib import Path
import sys
import time
import torch.distributed as dist
import os


_LOGGING_SRCFILE = logging._srcfile
_THIS_SRCFILE = os.path.normcase(__file__)


def _caller_class_name() -> str:
    # Avoid materializing frame.f_locals unless the first argument is `self`/`cls`.
    # `co_varnames` is a static attribute on the code object, reading it is free.
    frame = sys._getframe()
    while frame is not None:
        co_file = frame.f_code.co_filename
        if co_file != _LOGGING_SRCFILE and co_file != _THIS_SRCFILE:
            varnames = frame.f_code.co_varnames
            if varnames:
                first = varnames[0]
                if first == "self":
                    return type(frame.f_locals["self"]).__name__
                if first == "cls":
                    return frame.f_locals["cls"].__name__
            return ""
        frame = frame.f_back
    return ""


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


def _add_extra_fields_to_record():
    factory = logging.getLogRecordFactory()
    cwd = os.getcwd()

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.rank = _current_process_rank()
        try:
            record.relpath = os.path.relpath(record.pathname, cwd)
        except ValueError:
            # different drive on Windows, fall back to file name
            record.relpath = record.filename
        class_name = _caller_class_name()
        record.classname = class_name
        record.qualname = (
            f"{class_name}.{record.funcName}" if class_name else record.funcName
        )
        return record

    logging.setLogRecordFactory(record_factory)


def _configure_logging(
    logfile: Path | str,
    logging_level=logging.INFO,
):
    """
    Hydra installs its own logging handlers before main() runs.
    This function forcefully replaces them so our file + format take effect.
    """
    _add_extra_fields_to_record()

    fmt = logging.Formatter(
        "[rank:{rank}] [{levelname}] [{asctime}] [{relpath}:{lineno}] {qualname} : {message}",
        style="{",
    )

    file_handler = logging.FileHandler(logfile, mode="a")
    stream_handler = logging.StreamHandler()

    for handler in [file_handler, stream_handler]:
        handler.setLevel(logging_level)
        handler.setFormatter(fmt)

    # Replace existing handlers (Hydra already configured them)
    root = logging.getLogger()
    root.setLevel(logging_level)
    root.handlers[:] = [file_handler, stream_handler]


def string_to_logging_level(
    level_str: str,
) -> int:
    """
    Convert a string logging level to the corresponding logging module constant.
    """
    level_str = level_str.upper()
    if hasattr(logging, level_str):
        return getattr(logging, level_str)
    raise ValueError(f"Invalid logging level: {level_str}")