
# =============================================================================
# File: utils/time_utils.py
# =============================================================================
"""Time utilities."""

import time
from datetime import datetime
from typing import Optional


class FPSCounter:
    """Calculate frames per second."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """Update FPS counter and return current FPS."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) < 2:
            return 0.0
        
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed > 0:
            return (len(self.frame_times) - 1) / elapsed
        return 0.0


def get_timestamp() -> str:
    """Get ISO format timestamp."""
    return datetime.now().isoformat()