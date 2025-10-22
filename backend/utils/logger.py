"""Logging utilities for the application."""

import sys
from pathlib import Path
from loguru import logger
from config import settings


def setup_logger():
    """Configure application-wide logging."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # File handler
    log_path = Path(settings.log_file)
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="100 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info(f"Logger initialized - Level: {settings.log_level}, File: {log_path}")
    
    return logger


# Initialize logger
app_logger = setup_logger()


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)
