
# =============================================================================
# File: relay/relay_simulator.py
# =============================================================================
"""Simulated relay for testing."""

import time
from typing import Dict
from .relay_interface import RelayInterface
from utils.logger import get_logger

logger = get_logger("RelaySimulator")


class RelaySimulator(RelayInterface):
    """Simulated relay controller."""
    
    def __init__(self):
        """Initialize simulator."""
        self.states: Dict[int, bool] = {}
        logger.info("Relay simulator initialized")
    
    def activate(self, relay_id: int, duration: float = 1.0) -> bool:
        """Activate a relay."""
        self.states[relay_id] = True
        logger.info(f"RELAY {relay_id} ACTIVATED (simulated) for {duration}s")
        return True
    
    def deactivate(self, relay_id: int) -> bool:
        """Deactivate a relay."""
        self.states[relay_id] = False
        logger.info(f"RELAY {relay_id} DEACTIVATED (simulated)")
        return True
    
    def get_state(self, relay_id: int) -> bool:
        """Get relay state."""
        return self.states.get(relay_id, False)