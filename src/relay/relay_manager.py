
# =============================================================================
# File: relay/relay_manager.py
# =============================================================================
"""Relay orchestration and cooldown management."""

import time
import threading
from typing import Dict, Optional
from .relay_interface import RelayInterface
from .relay_simulator import RelaySimulator
from utils.logger import get_logger

logger = get_logger("RelayManager")


class RelayManager:
    """Manage relay activation with cooldown."""
    
    def __init__(
        self,
        interface: Optional[RelayInterface] = None,
        cooldown: float = 5.0,
        activation_duration: float = 1.0
    ):
        """Initialize relay manager.
        
        Args:
            interface: Relay hardware interface (None = simulator)
            cooldown: Minimum time between activations (seconds)
            activation_duration: How long to keep relay active (seconds)
        """
        self.interface = interface or RelaySimulator()
        self.cooldown = cooldown
        self.activation_duration = activation_duration
        
        self.last_activation: Dict[int, float] = {}
        self.lock = threading.Lock()
    
    def trigger(self, relay_id: int) -> bool:
        """Trigger a relay if cooldown has elapsed.
        
        Args:
            relay_id: Relay channel number
        
        Returns:
            True if relay was triggered
        """
        with self.lock:
            current_time = time.time()
            last_time = self.last_activation.get(relay_id, 0)
            
            # Check cooldown
            if current_time - last_time < self.cooldown:
                logger.debug(
                    f"Relay {relay_id} in cooldown "
                    f"({current_time - last_time:.1f}s / {self.cooldown}s)"
                )
                return False
            
            # Activate relay
            success = self.interface.activate(relay_id, self.activation_duration)
            
            if success:
                self.last_activation[relay_id] = current_time
                
                # Schedule deactivation
                timer = threading.Timer(
                    self.activation_duration,
                    self._deactivate_relay,
                    args=(relay_id,)
                )
                timer.daemon = True
                timer.start()
            
            return success
    
    def _deactivate_relay(self, relay_id: int) -> None:
        """Deactivate relay after duration."""
        self.interface.deactivate(relay_id)
    
    def get_state(self, relay_id: int) -> bool:
        """Get relay state."""
        return self.interface.get_state(relay_id)
    
    def is_in_cooldown(self, relay_id: int) -> bool:
        """Check if relay is in cooldown period."""
        with self.lock:
            current_time = time.time()
            last_time = self.last_activation.get(relay_id, 0)
            return (current_time - last_time) < self.cooldown
