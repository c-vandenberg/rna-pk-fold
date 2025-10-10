import logging
import sys
import time
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
    Generates a standardized file path for a log file.

    This function creates a safe filename from a module name and appends a
    timestamp to ensure uniqueness. It also ensures the target log directory exists.

    Parameters
    ----------
    module_name : str
        The name of the module or logger (e.g., "my_app.utils").
    log_dir : Optional[Path], optional
        The directory where the log file will be saved. Defaults to `DEFAULT_LOG_DIR`.
    include_timestamp : bool, optional
        If True, a timestamp is added to the filename to prevent overwrites,
        by default True.

    Returns
    -------
    Path
        The full `pathlib.Path` object for the generated log file.
    """
    # Use the default log directory if none is provided.
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    # Create the log directory and any necessary parent directories.
    # `exist_ok=True` prevents an error if the directory already exists.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize the module name to create a filesystem-safe filename.
    safe_name = module_name.replace(".", "_")

    # Append a timestamp to the filename for uniqueness if requested.
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.log"
    else:
        filename = f"{safe_name}.log"

    # Combine the directory and filename into a full Path object.
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
    """
    Configures and returns a logger with console and optional file handlers.

    This function provides a standardized way to set up logging for different
    modules. It clears any existing handlers to prevent duplicate messages and
    configures new handlers for console (stdout) and file output with distinct
    log levels if needed.

    Parameters
    ----------
    name : str
        The name of the logger, typically `__name__`.
    level : int, optional
        The base logging level for the logger and its handlers, by default `logging.INFO`.
    log_file : Optional[str], optional
        A specific path for the log file. If provided, it overrides the
        automatic path generation.
    log_dir : Optional[Path], optional
        The directory to store the log file if `log_file` is not provided.
        Defaults to `DEFAULT_LOG_DIR`.
    enable_file_logging : bool, optional
        If True and `log_file` is not specified, a default timestamped log file
        will be created. If False, no file logging will occur unless `log_file`
        is explicitly set. By default True.
    console_level : Optional[int], optional
        An override for the logging level of the console handler. If None, it
        defaults to the base `level`.
    file_level : Optional[int], optional
        An override for the logging level of the file handler. If None, it
        defaults to the base `level`.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level if console_level is not None else level)
    logger.addHandler(console_handler)

    file_handler = None
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
    elif enable_file_logging:
        log_path = get_log_file_path(name, log_dir=log_dir, include_timestamp=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        logger.info(f"Logging to file: {log_path}")

    if file_handler:
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level if file_level is not None else level)
        logger.addHandler(file_handler)

    return logger


def set_log_level(logger: logging.Logger, level: int) -> None:
    """
    Dynamically updates the logging level for a logger and all its handlers.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to update.
    level : int
        The new logging level to set (e.g., `logging.DEBUG`, `logging.INFO`).
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def cleanup_old_logs(log_dir: Optional[Path] = None, days_to_keep: int = 7) -> None:
    """
    Removes log files from a directory that are older than a specified number of days.

    This function helps manage disk space by automatically deleting old logs based on
    their last modification time.

    Parameters
    ----------
    log_dir : Optional[Path], optional
        The directory to clean. Defaults to `DEFAULT_LOG_DIR`.
    days_to_keep : int, optional
        The maximum age of log files to keep, in days. By default 7.
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    if not log_dir.exists():
        return

    cutoff_time = time.time() - (days_to_keep * 86400)

    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            print(f"Removed old log: {log_file}")
