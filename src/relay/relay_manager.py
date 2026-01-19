
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
        self.active_timers: Dict[int, threading.Timer] = {}  # Track pending deactivations
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
                
                # Cancel any previous pending timer for this relay
                if relay_id in self.active_timers:
                    old_timer = self.active_timers[relay_id]
                    if old_timer.is_alive():
                        old_timer.cancel()
                
                # Schedule deactivation (tracked for proper cleanup)
                timer = threading.Timer(
                    self.activation_duration,
                    self._deactivate_relay,
                    args=(relay_id,)
                )
                timer.daemon = True
                self.active_timers[relay_id] = timer  # Track for cleanup
                timer.start()
            
            return success
    
    def _deactivate_relay(self, relay_id: int) -> None:
        """Deactivate relay after duration."""
        try:
            self.interface.deactivate(relay_id)
        finally:
            # Always remove from tracking
            with self.lock:
                if relay_id in self.active_timers:
                    del self.active_timers[relay_id]
    
    def get_state(self, relay_id: int) -> bool:
        """Get relay state."""
        return self.interface.get_state(relay_id)
    
    def is_in_cooldown(self, relay_id: int) -> bool:
        """Check if relay is in cooldown period."""
        with self.lock:
            current_time = time.time()
            last_time = self.last_activation.get(relay_id, 0)
            return (current_time - last_time) < self.cooldown
    
    def shutdown(self) -> None:
        """Shutdown relay manager - cancel all pending timers and deactivate relays.
        
        CRITICAL: This prevents relays from remaining stuck active if app crashes.
        """
        logger.info("RelayManager: Starting shutdown...")
        
        with self.lock:
            # Cancel all pending timers
            for relay_id, timer in list(self.active_timers.items()):
                try:
                    if timer.is_alive():
                        timer.cancel()
                        logger.debug(f"  Cancelled timer for relay {relay_id}")
                except Exception as e:
                    logger.warning(f"Error cancelling timer for relay {relay_id}: {e}")
            
            self.active_timers.clear()
        
        # Deactivate all relays
        try:
            self.interface.shutdown()
            logger.info("  OK: All relays deactivated")
        except Exception as e:
            logger.warning(f"Error during relay shutdown: {e}")
        
        logger.info("  OK: RelayManager shutdown complete")
