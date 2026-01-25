
# =============================================================================
# File: relay/relay_manager.py
# =============================================================================
"""
Flexible Relay Management System

Supports 2, 3, 4, 5, 6, 8, or 16 channel USB relays with auto-detection.
No hardcoding - works with any number of available channels.
"""

import os
import time
import threading
import logging
import pathlib
from typing import Optional, Dict, List

try:
    import pyhid_usb_relay
except ImportError:
    pyhid_usb_relay = None

from utils.logger import get_logger

logger = get_logger("RelayManager")

# Workaround for pyhid_usb_relay DLL path issue on Windows
def _create_working_find():
    """Create a working find() function that handles the libusb DLL path correctly."""
    if pyhid_usb_relay is None:
        return None
    
    def find_relay_device(find_all=False, serial=None, bus=None, address=None):
        """Find USB relay device(s), handling the libusb DLL path correctly."""
        import usb.core
        import usb.util
        import usb.backend.libusb1
        import libusb
        
        from pyhid_usb_relay.controller import Controller
        
        VENDOR_ID = 0x16C0
        PRODUCT_ID = 0x05DF
        
        # Get the libusb DLL path - try multiple locations
        dll_path = None
        try:
            # Try x86_64 path first (Windows 64-bit)
            dlls = list(pathlib.Path(libusb.__file__).parent.rglob("x86_64/libusb-1.0.dll"))
            if dlls:
                dll_path = str(dlls[0])
            else:
                # Try arm64 path
                dlls = list(pathlib.Path(libusb.__file__).parent.rglob("arm64/libusb-1.0.dll"))
                if dlls:
                    dll_path = str(dlls[0])
                else:
                    # Try generic search
                    dlls = list(pathlib.Path(libusb.__file__).parent.rglob("*/libusb-1.0.dll"))
                    if dlls:
                        dll_path = str(dlls[0])
        except Exception as e:
            logger.warning(f"Could not find libusb DLL: {e}")
            return None
        
        if not dll_path:
            logger.warning("No libusb DLL found in any location")
            return None
        
        logger.debug(f"Using libusb DLL: {dll_path}")
        
        # Create backend with correct DLL path
        try:
            backend = usb.backend.libusb1.get_backend(find_library=lambda x: dll_path)
            if backend is None:
                logger.warning("Could not create libusb backend")
                return None
        except Exception as e:
            logger.warning(f"Failed to create backend: {e}")
            return None
        
        # Match function for relay devices
        def match_relay(device):
            try:
                manufacturer = usb.util.get_string(device, device.iManufacturer)
                product = usb.util.get_string(device, device.iProduct)
                
                if manufacturer != "www.dcttech.com":
                    return False
                if not product.startswith("USBRelay"):
                    return False
                
                if serial is not None:
                    try:
                        c = Controller(device)
                        if c.serial != serial:
                            return False
                    except Exception:
                        return False
                
                if bus is not None and device.bus != bus:
                    return False
                if address is not None and device.address != address:
                    return False
                
                return True
            except Exception as e:
                logger.debug(f"Match check failed: {e}")
                return False
        
        # Find devices
        try:
            devices = usb.core.find(
                backend=backend,
                find_all=find_all,
                idVendor=VENDOR_ID,
                idProduct=PRODUCT_ID,
                custom_match=match_relay,
            )
            
            if devices is None:
                logger.debug("No relay devices found")
                return None
            
            # Handle single vs multiple devices
            if find_all:
                if isinstance(devices, list):
                    return [Controller(d) for d in devices] if devices else None
                else:
                    return [Controller(devices)] if devices else None
            else:
                # Single device mode
                if isinstance(devices, list) and devices:
                    return Controller(devices[0])
                elif not isinstance(devices, list):
                    return Controller(devices)
                else:
                    return None
                    
        except Exception as e:
            logger.warning(f"Failed to find relay: {e}")
            return None
    
    return find_relay_device

# Replace pyhid_usb_relay.find with our working version
if pyhid_usb_relay is not None:
    working_find = _create_working_find()
    if working_find:
        pyhid_usb_relay.find = working_find
        logger.debug("Installed working relay find() function")


class RelayManager:
    """
    Flexible relay manager supporting any number of channels (2-16).
    
    Auto-detects relay device and available channels.
    No hardcoding - works with whatever relay is connected.
    
    Simple state management for problem/ok indicator logic.
    """
    
    def __init__(
        self,
        cooldown: float = 0.5,
        activation_duration: float = 1.0
    ):
        """Initialize relay manager.
        
        Args:
            cooldown: Minimum time between activations (seconds)
            activation_duration: How long to keep relay active (seconds)
        """
        self.cooldown = cooldown
        self.activation_duration = activation_duration
        
        # Relay device
        self.device = None
        self.num_channels = 0
        self.available_channels: List[int] = []
        
        # State tracking
        self.channel_states: Dict[int, bool] = {}
        self.last_activation: Dict[int, float] = {}
        self.active_timers: Dict[int, threading.Timer] = {}
        self.lock = threading.Lock()
        
        # Auto-detect on init
        self._detect_relay()
        
        logger.info(
            f"RelayManager initialized: "
            f"available_channels={self.available_channels}, "
            f"cooldown={cooldown}s, duration={activation_duration}s"
        )
    
    def _detect_relay(self) -> None:
        """Auto-detect relay device and available channels."""
        if pyhid_usb_relay is None:
            logger.warning("pyhid_usb_relay not available - relay disabled")
            self.device = None
            self.num_channels = 0
            return
        
        try:
            logger.debug("Attempting to find relay device...")
            # Use original pyhid_usb_relay.find() (now with patched backend)
            self.device = pyhid_usb_relay.find()
            if self.device is None:
                logger.warning("No relay device found via pyhid_usb_relay.find()")
                return
            
            logger.debug(f"Device found: {self.device}")
            
            # Auto-detect number of channels by testing each one
            # Read device state to check which channels exist
            max_channels = 16  # Maximum possible
            detected_channels = []
            
            # Get current state to understand the device
            try:
                initial_state = self.device.state
                logger.debug(f"Relay device state: {initial_state}")
            except Exception as e:
                logger.warning(f"Could not read initial state: {e}")
            
            for ch in range(1, max_channels + 1):
                try:
                    # Try to toggle the channel to test if it exists
                    self.device.toggle_state(ch)
                    detected_channels.append(ch)
                    time.sleep(0.05)
                    # Toggle back off
                    self.device.toggle_state(ch)
                    time.sleep(0.05)
                except Exception as e:
                    # Channel doesn't exist, stop trying
                    logger.debug(f"Channel {ch} test failed: {e}")
                    break
            
            self.num_channels = len(detected_channels)
            self.available_channels = detected_channels
            
            # Initialize state tracking for available channels
            with self.lock:
                for ch in self.available_channels:
                    self.channel_states[ch] = False
            
            logger.info(
                f"[OK] Relay device detected with {self.num_channels} channels: {detected_channels}"
            )
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to detect relay device: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self.device = None
            self.num_channels = 0
    
    def set_state(self, channel: int, state: bool, reason: str = "") -> bool:
        """Set relay channel state.
        
        Args:
            channel: Relay channel number
            state: True = ON, False = OFF
            reason: Reason for state change (for logging)
        
        Returns:
            True if successful, False if in cooldown or invalid channel
        """
        # Validate channel
        if channel not in self.available_channels:
            logger.error(f"Invalid relay channel: {channel} (available: {self.available_channels})")
            return False
        
        # Check if device available
        if self.device is None:
            logger.debug(f"Relay channel {channel} set requested but no device connected")
            return False
        
        with self.lock:
            old_state = self.channel_states.get(channel, False)
            
            # If setting to False, just turn off
            if not state:
                try:
                    self._set_relay_internal(channel, False)
                    if old_state != state:
                        logger.info(f"Relay {channel} -> OFF{f' ({reason})' if reason else ''}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to set relay {channel} off: {e}")
                    return False
            
            # If setting to True, check cooldown
            current_time = time.time()
            last_time = self.last_activation.get(channel, 0)
            
            if current_time - last_time < self.cooldown:
                logger.debug(
                    f"Relay {channel} in cooldown ({current_time - last_time:.1f}s / {self.cooldown}s)"
                )
                return False
            
            # Activate relay
            try:
                self._set_relay_internal(channel, True)
                self.last_activation[channel] = current_time
                logger.info(f"Relay {channel} -> ON{f' ({reason})' if reason else ''}")
                
                # Schedule deactivation
                if channel in self.active_timers:
                    old_timer = self.active_timers[channel]
                    if old_timer.is_alive():
                        old_timer.cancel()
                
                timer = threading.Timer(
                    self.activation_duration,
                    self._deactivate_relay,
                    args=(channel,)
                )
                timer.daemon = True
                self.active_timers[channel] = timer
                timer.start()
                
                return True
            except Exception as e:
                logger.error(f"Failed to set relay {channel} on: {e}")
                return False
    
    def get_state(self, channel: int) -> Optional[bool]:
        """Get relay channel state.
        
        Args:
            channel: Relay channel number
        
        Returns:
            True/False if valid channel, None if invalid
        """
        if channel not in self.available_channels:
            return None
        
        with self.lock:
            return self.channel_states.get(channel, False)
    
    def get_status(self) -> Dict:
        """Get current relay status.
        
        Returns:
            Dictionary with status of all available channels
        """
        with self.lock:
            states = {f'relay_{ch}': self.channel_states.get(ch, False) 
                     for ch in self.available_channels}
            
            return {
                **states,
                'num_channels': self.num_channels,
                'available_channels': self.available_channels,
                'available': self.device is not None,
                'any_active': any(self.channel_states.get(ch, False) 
                                  for ch in self.available_channels)
            }
    
    def test_relay(self, channel: int, duration: float = 1.0) -> bool:
        """Test a single relay channel.
        
        IMPORTANT: This test assumes toggle_state() works correctly.
        If device.state is cached/stale, we rely on timing to ensure
        the physical relay reaches the correct state.
        
        Args:
            channel: Relay channel to test
            duration: Test duration in seconds
        
        Returns:
            True if test successful, False otherwise
        """
        if channel not in self.available_channels:
            logger.error(f"Invalid test channel: {channel} (available: {self.available_channels})")
            return False
        
        if self.device is None:
            logger.warning("Cannot test relay - no device connected")
            return False
        
        try:
            logger.info(f"Testing relay channel {channel}...")
            
            with self.lock:
                # Step 1: Ensure relay is OFF before test
                # Read current state (may be cached)
                try:
                    current_state = self.device.state
                    is_on = bool(current_state & (1 << (channel - 1)))
                    if is_on:
                        logger.debug(f"Test {channel}: Initial state is ON, toggling OFF first")
                        self.device.toggle_state(channel)
                        time.sleep(0.2)
                except Exception:
                    pass  # If state read fails, just proceed
                
                # Step 2: Toggle ON for test
                logger.debug(f"Test {channel}: Toggling ON")
                self.channel_states[channel] = True
                self.device.toggle_state(channel)
                time.sleep(0.2)  # Wait for device to respond to toggle
            
            # Keep ON for test duration
            time.sleep(duration)
            
            # Step 3: Toggle OFF (just toggle once, don't verify - avoids cached state issue)
            with self.lock:
                logger.debug(f"Test {channel}: Toggling OFF")
                self.channel_states[channel] = False
                self.device.toggle_state(channel)
                time.sleep(0.3)  # Wait for device to respond
            
            logger.info(f"[OK] Relay {channel} test successful")
            return True
        except Exception as e:
            logger.error(f"Relay {channel} test failed: {e}")
            return False
    
    def test_all_relays(self, duration: float = 1.0, interval: float = 1.0) -> Dict[int, bool]:
        """Test all relay channels.
        
        Args:
            duration: Test duration per relay in seconds
            interval: Interval between tests in seconds
        
        Returns:
            Dictionary with test results for each channel
        """
        results = {}
        
        for channel in self.available_channels:
            results[channel] = self.test_relay(channel, duration)
            if channel != self.available_channels[-1]:  # Don't sleep after last test
                time.sleep(interval)
        
        logger.info(f"All relay tests complete: {results}")
        return results
    
    def safe_off(self) -> None:
        """Turn all relays OFF immediately (emergency stop).
        
        CRITICAL: This uses toggle-then-verify approach to handle cached device.state.
        We toggle OFF and verify, then toggle again if still ON.
        """
        if not self.device or not self.available_channels:
            logger.debug("No relay device or channels - skipping safe off")
            return
        
        try:
            with self.lock:
                logger.debug(f"Safe OFF: Starting shutdown sequence for {len(self.available_channels)} relays")
                
                # Attempt 1: Toggle all relays OFF
                for ch in self.available_channels:
                    self.channel_states[ch] = False
                    try:
                        logger.debug(f"Safe OFF: Toggle OFF relay {ch}")
                        self.device.toggle_state(ch)
                        time.sleep(0.1)  # Wait for toggle to complete
                    except Exception as e:
                        logger.error(f"Failed to toggle relay {ch}: {e}")
                
                # Wait a bit for device to update
                time.sleep(0.2)
                
                # Verify all relays are OFF
                try:
                    final_state = self.device.state
                    logger.debug(f"Safe OFF: Device state after toggles = {bin(final_state)}")
                    
                    # Check if any relay is still ON
                    still_on = []
                    for ch in self.available_channels:
                        is_on = bool(final_state & (1 << (ch - 1)))
                        if is_on:
                            still_on.append(ch)
                    
                    # If any relay is still ON, toggle them again
                    if still_on:
                        logger.warning(f"Safe OFF: Relays still ON: {still_on}, toggling again")
                        for ch in still_on:
                            try:
                                logger.debug(f"Safe OFF: Second attempt - Toggle OFF relay {ch}")
                                self.device.toggle_state(ch)
                                time.sleep(0.1)
                            except Exception as e:
                                logger.error(f"Failed to toggle relay {ch} (2nd attempt): {e}")
                        
                        time.sleep(0.2)
                        final_state = self.device.state
                        logger.debug(f"Safe OFF: Final device state = {bin(final_state)}")
                    
                except Exception as e:
                    logger.warning(f"Could not verify final state: {e}")
            
            logger.warning("All relays turned OFF (safe off)")
        except Exception as e:
            logger.error(f"Error during safe off: {e}")
    
    def trigger(self, channel: int) -> bool:
        """Trigger a relay (backward compatible).
        
        Args:
            channel: Relay channel
        
        Returns:
            True if triggered
        """
        return self.set_state(channel, True, reason="Legacy trigger")
    
    # ========== PRIVATE METHODS ==========
    
    def _set_relay_internal(self, channel: int, state: bool) -> None:
        """Internal method to set relay state without lock."""
        self.channel_states[channel] = state
        
        if self.device:
            try:
                # Try to read the ACTUAL physical state from the device
                time.sleep(0.01)  # Small delay to ensure device state is fresh
                current_state = self.device.state
                
                # Check if relay is currently in the opposite state
                is_on = bool(current_state & (1 << (channel - 1)))
                want_on = bool(state)
                
                logger.debug(f"Relay {channel}: current={'ON' if is_on else 'OFF'}, want={'ON' if want_on else 'OFF'}")
                
                # Only toggle if state doesn't match
                if is_on != want_on:
                    logger.debug(f"Relay {channel}: Toggling from {is_on} to {want_on}")
                    self.device.toggle_state(channel)
                    time.sleep(0.05)  # Wait for toggle to complete
                else:
                    logger.debug(f"Relay {channel}: Already in correct state, no toggle")
                    
            except Exception as e:
                logger.error(f"Failed to set relay {channel} physical state: {e}")
        else:
            state_str = "ON" if state else "OFF"
            logger.debug(f"[SIM] Relay {channel} -> {state_str}")
    
    def _deactivate_relay(self, channel: int) -> None:
        """Deactivate relay after duration."""
        try:
            with self.lock:
                self._set_relay_internal(channel, False)
        finally:
            with self.lock:
                if channel in self.active_timers:
                    del self.active_timers[channel]
    
    def shutdown(self) -> None:
        """Shutdown relay manager and cancel all timers."""
        logger.info("RelayManager: Starting shutdown...")
        
        with self.lock:
            # Cancel all pending timers
            for channel, timer in list(self.active_timers.items()):
                try:
                    if timer.is_alive():
                        timer.cancel()
                        logger.debug(f"  Cancelled timer for relay {channel}")
                except Exception as e:
                    logger.warning(f"Error cancelling timer for relay {channel}: {e}")
            
            self.active_timers.clear()
        
        # Safe off all relays
        try:
            self.safe_off()
        except Exception as e:
            logger.warning(f"Error during shutdown safe-off: {e}")
        
        logger.info("  OK: RelayManager shutdown complete")
