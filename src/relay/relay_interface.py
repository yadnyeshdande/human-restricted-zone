
# =============================================================================
# File: relay/relay_interface.py
# =============================================================================
"""Hardware abstraction layer for relays."""

from abc import ABC, abstractmethod
from utils.logger import get_logger

logger = get_logger("RelayInterface")


class RelayInterface(ABC):
    """Abstract base class for relay control."""
    
    @abstractmethod
    def activate(self, relay_id: int, duration: float = 1.0) -> bool:
        """Activate a relay for specified duration.
        
        Args:
            relay_id: Relay channel number
            duration: Activation duration in seconds
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def deactivate(self, relay_id: int) -> bool:
        """Deactivate a relay.
        
        Args:
            relay_id: Relay channel number
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_state(self, relay_id: int) -> bool:
        """Get relay state.
        
        Args:
            relay_id: Relay channel number
        
        Returns:
            True if relay is active
        """
        pass

