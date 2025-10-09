import logging
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Default log directory
DEFAULT_LOG_DIR = Path("var/log")


def get_log_file_path(
        module_name: str,
        log_dir: Optional[Path] = None,
        include_timestamp: bool = True
) -> Path:
    """
    Generate a log file path for a module.

    Parameters
    ----------
    module_name : str
        Name of the module (e.g., "rna_pk_fold.folding.eddy_rivas")
    log_dir : Path, optional
        Directory to store logs. Defaults to var/log
    include_timestamp : bool
        If True, add timestamp to filename

    Returns
    -------
    Path
        Full path to the log file
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    # Create directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clean module name for filename
    safe_name = module_name.replace(".", "_")

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.log"
    else:
        filename = f"{safe_name}.log"

    return log_dir / filename


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_tqdm: bool = True,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level if console_level is not None else level)
    logger.addHandler(console_handler)

    # File
    file_handler = None
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
    elif enable_file_logging:
        log_path = get_log_file_path(name, log_dir=log_dir, include_timestamp=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        # this INFO goes to console; fine — it tells you the exact path
        logger.info(f"Logging to file: {log_path}")

    if file_handler:
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level if file_level is not None else level)
        logger.addHandler(file_handler)

    return logger


def set_log_level(logger: logging.Logger, level: int) -> None:
    """Update logger and all its handlers to the new level."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def cleanup_old_logs(log_dir: Optional[Path] = None, days_to_keep: int = 7) -> None:
    """
    Remove log files older than specified days.

    Parameters
    ----------
    log_dir : Path, optional
        Log directory to clean (defaults to var/log)
    days_to_keep : int
        Keep logs from the last N days
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    if not log_dir.exists():
        return

    import time
    cutoff_time = time.time() - (days_to_keep * 86400)

    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            print(f"Removed old log: {log_file}")