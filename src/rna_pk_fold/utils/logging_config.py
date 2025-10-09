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
        enable_tqdm: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        If provided, use this exact log file path
    log_dir : Path, optional
        Directory for logs (default: var/log). Ignored if log_file is provided.
    enable_file_logging : bool
        If True and log_file is None, create a default log file
    enable_tqdm : bool
        If True, use handlers compatible with progress bars
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (compatible with tqdm)
    if enable_tqdm:
        try:
            from tqdm.contrib.logging import logging_redirect_tqdm
            console_handler = logging.StreamHandler(sys.stdout)
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
    else:
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Use explicit file path
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
    elif enable_file_logging:
        # Use default log directory
        log_path = get_log_file_path(name, log_dir=log_dir, include_timestamp=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        logger.info(f"Logging to file: {log_path}")
    else:
        file_handler = None

    if file_handler:
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_module_logger(
        module_name: str,
        level: int = logging.INFO,
        log_dir: Optional[Path] = None,
        enable_file_logging: bool = True
) -> logging.Logger:
    """
    Convenience function to set up a logger for a module with default settings.

    Parameters
    ----------
    module_name : str
        Module name (e.g., "rna_pk_fold.folding.zucker")
    level : int
        Logging level
    log_dir : Path, optional
        Log directory (defaults to var/log)
    enable_file_logging : bool
        Whether to log to file

    Returns
    -------
    logging.Logger
    """
    return setup_logger(
        name=module_name,
        level=level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_tqdm=True
    )


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