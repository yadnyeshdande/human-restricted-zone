
# =============================================================================
# File: utils/logger.py
# =============================================================================
"""Centralized logging with rotating file handler."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class SystemLogger:
    """Thread-safe centralized logging system."""
    
    _instance: Optional['SystemLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not SystemLogger._initialized:
            self._setup_logging()
            SystemLogger._initialized = True
    
    def _setup_logging(self) -> None:
        """Configure logging with rotating file handler."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Root logger
        self.logger = logging.getLogger("VisionSafety")
        self.logger.setLevel(logging.DEBUG)
        
        # Rotating file handler (10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "vision_safety.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a child logger."""
        return self.logger.getChild(name)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return SystemLogger().get_logger(name)
