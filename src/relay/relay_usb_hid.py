# =============================================================================
# ADDITIONAL FILE: relay/relay_usb_hid.py (ADD THIS NEW FILE)
# =============================================================================
"""USB HID relay controller using pyhid_usb_relay library."""

import time
from typing import Dict, Optional
from .relay_interface import RelayInterface
from utils.logger import get_logger

logger = get_logger("RelayUSBHID")


class RelayUSBHID(RelayInterface):
    """USB HID relay controller using pyhid_usb_relay library.
    
    Install: pip install pyhid_usb_relay
    
    Usage example:
        import pyhid_usb_relay
        relay = pyhid_usb_relay.find()
        relay.toggle_state(1)  # Toggle relay 1
    """
    
    def __init__(self, num_channels: int = 8, serial: Optional[str] = None):
        """Initialize USB HID relay.
        
        Args:
            num_channels: Number of relay channels (default: 8)
            serial: Optional serial number if multiple devices connected
        """
        self.num_channels = num_channels
        self.serial = serial
        self.device = None
        self.states: Dict[int, bool] = {}
        
        try:
            import pyhid_usb_relay
            self.relay_lib = pyhid_usb_relay
            self._connect()
        except ImportError:
            logger.error("pyhid_usb_relay not found. Install: pip install pyhid_usb_relay")
            raise
    
    def _connect(self) -> bool:
        """Connect to USB HID relay device."""
        try:
            # Find relay device
            if self.serial:
                # Connect to specific device by serial
                self.device = self.relay_lib.find(serial=self.serial)
            else:
                # Find first available device
                self.device = self.relay_lib.find()
            
            if not self.device:
                logger.error("No USB relay device found")
                return False
            
            logger.info(f"USB Relay connected: {self.num_channels} channels")
            logger.info(f"Current relay state: {self.device.state}")
            
            # Turn off all relays initially
            for i in range(1, self.num_channels + 1):
                self.deactivate(i)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect USB relay: {e}")
            logger.error("Make sure the device is connected and you have permissions")
            return False
    
    def activate(self, relay_id: int, duration: float = 1.0) -> bool:
        """Activate relay channel.
        
        Args:
            relay_id: Relay channel number (1-based)
            duration: Not used, managed by RelayManager
        
        Returns:
            True if successful
        """
        if not self.device:
            logger.error("USB relay device not connected")
            return False
        
        if relay_id < 1 or relay_id > self.num_channels:
            logger.error(f"Invalid relay ID {relay_id}. Valid range: 1-{self.num_channels}")
            return False
        
        try:
            # Get current state
            current_state = self.device.state
            
            # Check if relay is already on
            if current_state & (1 << (relay_id - 1)):
                logger.debug(f"Relay {relay_id} already ON")
                self.states[relay_id] = True
                return True
            
            # Turn on relay using toggle_state
            self.device.toggle_state(relay_id)
            self.states[relay_id] = True
            
            logger.info(f"USB RELAY {relay_id} ACTIVATED")
            logger.debug(f"New relay state: {self.device.state}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate relay {relay_id}: {e}")
            return False
    
    def deactivate(self, relay_id: int) -> bool:
        """Deactivate relay channel.
        
        Args:
            relay_id: Relay channel number (1-based)
        
        Returns:
            True if successful
        """
        if not self.device:
            logger.error("USB relay device not connected")
            return False
        
        if relay_id < 1 or relay_id > self.num_channels:
            logger.error(f"Invalid relay ID {relay_id}. Valid range: 1-{self.num_channels}")
            return False
        
        try:
            # Get current state
            current_state = self.device.state
            
            # Check if relay is already off
            if not (current_state & (1 << (relay_id - 1))):
                logger.debug(f"Relay {relay_id} already OFF")
                self.states[relay_id] = False
                return True
            
            # Turn off relay using toggle_state
            self.device.toggle_state(relay_id)
            self.states[relay_id] = False
            
            logger.info(f"USB RELAY {relay_id} DEACTIVATED")
            logger.debug(f"New relay state: {self.device.state}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate relay {relay_id}: {e}")
            return False
    
    def get_state(self, relay_id: int) -> bool:
        """Get relay state.
        
        Args:
            relay_id: Relay channel number (1-based)
        
        Returns:
            True if relay is active
        """
        if not self.device:
            return self.states.get(relay_id, False)
        
        try:
            # Read actual state from device
            current_state = self.device.state
            is_on = bool(current_state & (1 << (relay_id - 1)))
            self.states[relay_id] = is_on
            return is_on
        except Exception as e:
            logger.error(f"Failed to get relay {relay_id} state: {e}")
            return self.states.get(relay_id, False)
    
    def close(self) -> None:
        """Close USB connection and turn off all relays."""
        if self.device:
            logger.info("Closing USB relay connection...")
            try:
                # Turn off all relays
                for i in range(1, self.num_channels + 1):
                    self.deactivate(i)
                
                logger.info("USB relay disconnected, all channels OFF")
            except Exception as e:
                logger.error(f"Error while closing USB relay: {e}")
            finally:
                self.device = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()