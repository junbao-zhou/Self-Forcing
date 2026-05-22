

import logging
from importlib.metadata import distributions
from pathlib import Path
import sys
import time
import torch.distributed as dist
import os
import platform
import torch


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


def _install_excepthooks():
    """
    Route uncaught exceptions through the logging system so the traceback
    lands in the per-rank log file (and stderr handler), not just on stderr
    where torchrun may swallow it.
    """
    def _log_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            # Keep Ctrl-C behavior: don't spam logs, fall back to default.
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _log_uncaught

    # Python 3.8+: threading exceptions
    import threading
    def _log_thread_exc(args):
        if issubclass(args.exc_type, SystemExit):
            return
        logging.critical(
            f"Uncaught exception in thread {args.thread.name}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _log_thread_exc

    # Unraisable exceptions (e.g. errors in __del__, weakref callbacks)
    def _log_unraisable(unraisable):
        logging.error(
            f"Unraisable exception: {unraisable.err_msg or ''}",
            exc_info=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback),
        )
    sys.unraisablehook = _log_unraisable


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

    file_handler = logging.FileHandler(logfile, mode="w")
    stream_handler = logging.StreamHandler()

    for handler in [file_handler, stream_handler]:
        handler.setLevel(logging_level)
        handler.setFormatter(fmt)

    # Replace existing handlers (Hydra already configured them)
    root = logging.getLogger()
    root.setLevel(logging_level)
    root.handlers[:] = [file_handler, stream_handler]

    _install_excepthooks()


def _installed_package_versions() -> str:
    package_versions: dict[str, str] = {}
    for distribution in distributions():
        package_name = distribution.metadata.get("Name")
        if package_name is None:
            continue
        package_versions[package_name] = distribution.version

    return "\n".join(
        f"{package_name}=={package_versions[package_name]}"
        for package_name in sorted(package_versions, key=str.lower)
    )


def log_environment_versions() -> None:
    """
    Log runtime information and all installed Python package versions.
    """
    cuda_device_summary = "unavailable"
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        cuda_device_summary = (
            f"index={current_device}, "
            f"name={torch.cuda.get_device_name(current_device)}, "
            f"capability={torch.cuda.get_device_capability(current_device)}"
        )

    logging.info(
        f"""
[environment versions]
python={sys.version.split()[0]}, executable={sys.executable}
platform={platform.platform()}
torch={torch.__version__}, torch_cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}
cuda_available={torch.cuda.is_available()}, cuda_device={cuda_device_summary}
installed_packages:
{_installed_package_versions()}
"""
    )


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
