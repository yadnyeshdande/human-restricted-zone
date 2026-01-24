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
        backoff_factor: float = 2.0,
        max_attempts: int = 10
    ):
        """Initialize reconnect policy.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for each retry
            max_attempts: Maximum reconnection attempts before requiring manual reset
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.max_attempts = max_attempts
        self.current_delay = initial_delay
        self.attempt = 0
        self.is_exhausted = False
    
    def wait(self) -> None:
        """Wait before next reconnection attempt."""
        if self.is_exhausted:
            logger.warning(f"Reconnection exhausted - maximum attempts ({self.max_attempts}) reached")
            logger.warning(f"Please check camera URL and credentials, then restart application")
            time.sleep(self.max_delay)  # Wait at max delay
            return
        
        logger.info(f"Reconnect attempt {self.attempt + 1}/{self.max_attempts}, waiting {self.current_delay:.1f}s")
        time.sleep(self.current_delay)
        
        self.attempt += 1
        
        # Check if we've exhausted attempts
        if self.attempt >= self.max_attempts:
            self.is_exhausted = True
            logger.error(f"Reconnection exhausted after {self.max_attempts} attempts")
            logger.error(f"This usually means:")
            logger.error(f"  1) RTSP credentials are incorrect (check username:password format)")
            logger.error(f"  2) Camera IP address or port is wrong")
            logger.error(f"  3) Camera has blocked connections (may need restart or time to unblock)")
            return
        
        self.current_delay = min(
            self.current_delay * self.backoff_factor,
            self.max_delay
        )
    
    def reset(self) -> None:
        """Reset policy after successful connection."""
        self.current_delay = self.initial_delay
        self.attempt = 0
        self.is_exhausted = False

