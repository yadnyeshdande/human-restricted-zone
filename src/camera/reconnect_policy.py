# =============================================================================
# File: camera/reconnect_policy.py
# =============================================================================
"""Reconnection policy with exponential backoff."""

import time
from utils.logger import get_logger

logger = get_logger("ReconnectPolicy")


class ReconnectPolicy:
    """Exponential backoff reconnection policy."""
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        """Initialize reconnect policy.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for each retry
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = initial_delay
        self.attempt = 0
    
    def wait(self) -> None:
        """Wait before next reconnection attempt."""
        logger.info(f"Reconnect attempt {self.attempt + 1}, waiting {self.current_delay:.1f}s")
        time.sleep(self.current_delay)
        
        self.attempt += 1
        self.current_delay = min(
            self.current_delay * self.backoff_factor,
            self.max_delay
        )
    
    def reset(self) -> None:
        """Reset policy after successful connection."""
        self.current_delay = self.initial_delay
        self.attempt = 0

