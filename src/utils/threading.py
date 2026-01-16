
# =============================================================================
# File: utils/threading.py
# =============================================================================
"""Thread utilities and helpers."""

import threading
from typing import Callable, Optional


class StoppableThread(threading.Thread):
    """Thread with graceful stop mechanism."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.daemon = True
    
    def stop(self) -> None:
        """Signal thread to stop."""
        self._stop_event.set()
    
    def stopped(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for stop signal."""
        return self._stop_event.wait(timeout)