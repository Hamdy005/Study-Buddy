import os
import sys
import logging
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    Standard logging handler to intercept standard logging messages 
    and redirect them to Loguru.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    """
    Configures Loguru for console and file logging, and intercepts
    standard library logging (uvicorn, httpx, etc.).
    """
    os.makedirs("logs", exist_ok=True)

    # Remove default handlers
    logger.remove()

    # Log format specifications
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

    # Console Handler (colored)
    logger.add(
        sys.stdout,
        level="INFO",
        format=console_format,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # Rotating File Handler (5 MB per file, max 3 backups)
    logger.add(
        "logs/app.log",
        level="INFO",
        format=file_format,
        rotation="5 MB",
        retention=3,
        encoding="utf-8",
        enqueue=True,
    )

    # Intercept standard Python logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Ensure uvicorn, fastapi, watchfiles, and httpx use the intercept handler
    for log_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "watchfiles", "httpx"):
        mod_logger = logging.getLogger(log_name)
        mod_logger.handlers = [InterceptHandler()]
        if "watchfiles" in log_name:
            mod_logger.setLevel(logging.WARNING)

    return logger

__all__ = ["logger", "setup_logging", "InterceptHandler"]
