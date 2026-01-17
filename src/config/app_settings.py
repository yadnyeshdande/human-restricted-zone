# =============================================================================
# ADDITIONAL FILE: config/app_settings.py (ADD THIS NEW FILE)
# =============================================================================
"""Performance and detection settings - TWEAK THESE FOR YOUR NEEDS."""

from dataclasses import dataclass
from typing import Tuple, Optional
from multiprocessing.util import get_logger
from pathlib import Path
from typing import Tuple

from utils import logger
from utils.time_utils import get_timestamp


@dataclass
class AppSettings:
    """All tweakable application settings in one place."""
    
    # =========================================================================
    # PROCESSING RESOLUTION - Speed vs Quality tradeoff
    # =========================================================================
    # Lower = Faster, Less accurate
    # Higher = Slower, More accurate
    # Options: (640,360), (960,540), (1280,720), (1920,1080)
    processing_resolution: Tuple[int, int] = (1280, 720)
    
    # =========================================================================
    # YOLO MODEL - Choose based on your hardware
    # =========================================================================
    # yolov8n.pt = Fastest, lowest accuracy (DEFAULT)
    # yolov8s.pt = Fast, good accuracy
    # yolov8m.pt = Medium speed, better accuracy (RECOMMENDED for accuracy)
    # yolov8l.pt = Slow, high accuracy
    # yolov8x.pt = Slowest, highest accuracy
    yolo_model: str = "yolov8n.pt"
    
    # =========================================================================
    # DETECTION CONFIDENCE - How sure YOLO must be
    # =========================================================================
    # 0.3-0.4 = Very sensitive, may have false positives
    # 0.5 = Balanced (DEFAULT)
    # 0.6-0.7 = Conservative, may miss some people
    detection_confidence: float = 0.5
    
    # =========================================================================
    # ZONE VIOLATION MODE - How to trigger relay
    # =========================================================================
    # 'center' = Only trigger when person's CENTER is inside zone (DEFAULT)
    # 'overlap' = Trigger when person's bounding box OVERLAPS with zone
    violation_mode: str = 'center'  # 'center' or 'overlap'
    
    # =========================================================================
    # RELAY SETTINGS
    # =========================================================================
    relay_cooldown: float = 5.0  # Seconds between relay activations
    relay_duration: float = 1.0  # How long relay stays active
    
    # =========================================================================
    # USB RELAY HARDWARE (pyhid_usb_relay)
    # =========================================================================
    use_usb_relay: bool = False  # Set to True to use USB relay hardware
    usb_num_channels: int = 8  # Number of relay channels on your board
    usb_serial: str = None  # Serial number (optional, for multiple devices)

    
    # =========================================================================
    # FRAME QUEUE SETTINGS
    # =========================================================================
    frame_queue_size: int = 30  # Larger = more memory, smoother
    
    # =========================================================================
    # UI UPDATE RATE
    # =========================================================================
    ui_update_fps: int = 30  # UI refresh rate (lower = less CPU)

    def save(self) -> None:
        '''Save settings to file.'''
        import json
        from pathlib import Path
        from utils.time_utils import get_timestamp
        from utils.logger import get_logger  # ADD THIS LINE

        logger = get_logger("AppSettings")  # ADD THIS LINE (Remove logger.get_logger to only get_logger)

        settings_file = Path("app_settings.json")
        
        data = {
            "timestamp": get_timestamp(),
            "processing_resolution": list(self.processing_resolution),
            "yolo_model": self.yolo_model,
            "detection_confidence": self.detection_confidence,
            "violation_mode": self.violation_mode,
            "relay_cooldown": self.relay_cooldown,
            "relay_duration": self.relay_duration,
            "use_usb_relay": self.use_usb_relay,
            "usb_num_channels": self.usb_num_channels,
            "usb_serial": self.usb_serial,
            "frame_queue_size": self.frame_queue_size,
            "ui_update_fps": self.ui_update_fps
        }
        
        with open(settings_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Settings saved to {settings_file}")
    
    def load(self) -> None:
        '''Load settings from file.'''
        import json
        from pathlib import Path
        from utils.logger import get_logger  # ADD THIS LINE
    
        logger = get_logger("AppSettings")  # ADD THIS LINE
        
        settings_file = Path("app_settings.json")
        
        if not settings_file.exists():
            logger.info("No settings file found, using defaults")
            return
        
        try:
            with open(settings_file, 'r') as f:
                data = json.load(f)
            
            self.processing_resolution = tuple(data.get("processing_resolution", [1280, 720]))
            self.yolo_model = data.get("yolo_model", "yolov8n.pt")
            self.detection_confidence = data.get("detection_confidence", 0.5)
            self.violation_mode = data.get("violation_mode", "center")
            self.relay_cooldown = data.get("relay_cooldown", 5.0)
            self.relay_duration = data.get("relay_duration", 1.0)
            self.use_usb_relay = data.get("use_usb_relay", False)
            self.usb_num_channels = data.get("usb_num_channels", 8)
            self.usb_serial = data.get("usb_serial")
            self.frame_queue_size = data.get("frame_queue_size", 30)
            self.ui_update_fps = data.get("ui_update_fps", 30)
            
            logger.info("Settings loaded from file")
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
    
    def reset_to_defaults(self) -> None:
        '''Reset all settings to default values.'''
        from utils.logger import get_logger  # ADD THIS LINE
    
        logger = get_logger("AppSettings")  # ADD THIS LINE
        
        self.processing_resolution = (1280, 720)
        self.yolo_model = "yolov8n.pt"
        self.detection_confidence = 0.5
        self.violation_mode = "center"
        self.relay_cooldown = 5.0
        self.relay_duration = 1.0
        self.use_usb_relay = False
        self.usb_num_channels = 8
        self.usb_serial = None
        self.frame_queue_size = 30
        self.ui_update_fps = 30
        logger.info("Settings reset to defaults")
        SETTINGS.load()


# Default settings instance
SETTINGS = AppSettings()


# =============================================================================
# MODIFICATION 1: Update detection/detector.py
# REPLACE the __init__ method with this:
# =============================================================================

# FIND THIS in detection/detector.py:
"""
    def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.5):
"""

# REPLACE WITH:
"""
    def __init__(self, model_name: str = None, conf_threshold: float = None):
        from config.app_settings import SETTINGS
        
        self.conf_threshold = conf_threshold or SETTINGS.detection_confidence
        model_name = model_name or SETTINGS.yolo_model
        
        logger.info(f"Initializing YOLO detector: {model_name}, confidence: {self.conf_threshold}")
"""


# =============================================================================
# MODIFICATION 2: Update detection/geometry.py
# ADD this new function for overlap detection:
# =============================================================================

# ADD THIS FUNCTION to detection/geometry.py:
"""
def bbox_overlaps_rect(bbox: Tuple[int, int, int, int], rect: Tuple[int, int, int, int]) -> bool:
    '''Check if bounding box overlaps with rectangle.
    
    Args:
        bbox: (x1, y1, x2, y2) person bounding box
        rect: (x1, y1, x2, y2) restricted zone
    
    Returns:
        True if any overlap exists
    '''
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    rect_x1, rect_y1, rect_x2, rect_y2 = rect
    
    # No overlap if one rectangle is to the left/right/above/below the other
    if bbox_x2 < rect_x1 or bbox_x1 > rect_x2:
        return False
    if bbox_y2 < rect_y1 or bbox_y1 > rect_y2:
        return False
    
    return True
"""


# =============================================================================
# MODIFICATION 3: Update detection/detection_worker.py
# REPLACE the violation checking logic:
# =============================================================================

# FIND THIS in detection_worker.py run() method:
"""
                # Check violations
                for bbox in persons:
                    center = bbox_center(bbox)
                    
                    for zone_id, rect, relay_id in self.zones:
                        if point_in_rect(center, rect):
                            logger.warning(
                                f"VIOLATION: Camera {self.camera_id}, "
                                f"Zone {zone_id}, Relay {relay_id}"
                            )
                            self.on_violation(self.camera_id, zone_id, relay_id, frame)
"""

# REPLACE WITH:
"""
                # Check violations
                from config.app_settings import SETTINGS
                from .geometry import bbox_overlaps_rect
                
                for bbox in persons:
                    for zone_id, rect, relay_id in self.zones:
                        violation = False
                        
                        if SETTINGS.violation_mode == 'center':
                            # Only trigger when person's center is inside zone
                            center = bbox_center(bbox)
                            violation = point_in_rect(center, rect)
                        elif SETTINGS.violation_mode == 'overlap':
                            # Trigger when person's bbox overlaps with zone
                            violation = bbox_overlaps_rect(bbox, rect)
                        
                        if violation:
                            logger.warning(
                                f"VIOLATION [{SETTINGS.violation_mode} mode]: "
                                f"Camera {self.camera_id}, Zone {zone_id}, Relay {relay_id}"
                            )
                            self.on_violation(self.camera_id, zone_id, relay_id, frame)
                            break  # Only trigger once per person
"""


# =============================================================================
# MODIFICATION 4: Update app.py to use USB relay
# REPLACE the relay manager initialization:
# =============================================================================

# FIND THIS in app.py:
"""
        logger.info("Initializing relay manager...")
        relay_manager = RelayManager()
"""

# REPLACE WITH:
"""
        logger.info("Initializing relay manager...")
        from config.app_settings import SETTINGS
        
        # Choose relay interface based on settings
        relay_interface = None
        if SETTINGS.use_usb_relay:
            try:
                from relay.relay_usb_hid import RelayUSBHID
                relay_interface = RelayUSBHID(
                    vendor_id=SETTINGS.usb_vendor_id,
                    product_id=SETTINGS.usb_product_id,
                    num_channels=SETTINGS.usb_num_channels
                )
                logger.info("Using USB HID relay hardware")
            except Exception as e:
                logger.error(f"Failed to initialize USB relay: {e}")
                logger.info("Falling back to relay simulator")
        
        relay_manager = RelayManager(
            interface=relay_interface,
            cooldown=SETTINGS.relay_cooldown,
            activation_duration=SETTINGS.relay_duration
        )
"""


# =============================================================================
# MODIFICATION 5: Update config_manager.py to use settings
# REPLACE processing_resolution initialization:
# =============================================================================

# FIND THIS in config/config_manager.py __init__:
"""
    def load(self) -> AppConfig:
        # Load configuration from file.
        if not self.config_path.exists():
            logger.info("Config file not found, creating new configuration")
            self.config = AppConfig()
"""

# REPLACE WITH:
"""
    def load(self) -> AppConfig:
        # Load configuration from file.
        from config.app_settings import SETTINGS
        
        if not self.config_path.exists():
            logger.info("Config file not found, creating new configuration")
            self.config = AppConfig(processing_resolution=SETTINGS.processing_resolution)
"""


# =============================================================================
# EXPLANATION: HOW ZONE-TO-RELAY MAPPING WORKS
# =============================================================================
"""
ZONE TO RELAY ASSIGNMENT - SEQUENTIAL GLOBAL MAPPING:

The mapping is done in config_manager.py in the add_zone() method:

    def add_zone(self, camera_id: int, rect: Tuple[int, int, int, int]) -> Optional[Zone]:
        camera = self.get_camera(camera_id)
        if camera is None:
            return None
        
        zone = Zone(
            id=self._next_zone_id,      # Zone gets unique ID
            rect=rect,
            relay_id=self._next_relay_id  # ← RELAY ASSIGNMENT HAPPENS HERE
        )
        self._next_zone_id += 1
        self._next_relay_id += 1          # ← INCREMENT FOR NEXT ZONE
        
        camera.zones.append(zone)
        return zone

EXAMPLE MAPPING:
User draws zones in this order:
1. Camera 1, Zone 1 → Gets relay_id = 1 (self._next_relay_id starts at 1)
2. Camera 1, Zone 2 → Gets relay_id = 2 (incremented)
3. Camera 2, Zone 1 → Gets relay_id = 3 (incremented)
4. Camera 2, Zone 2 → Gets relay_id = 4 (incremented)
5. Camera 3, Zone 1 → Gets relay_id = 5 (incremented)

This mapping is saved in human_boundaries.json:
{
  "cameras": [
    {
      "id": 1,
      "zones": [
        {"id": 1, "rect": [...], "relay_id": 1},  ← Zone 1 → Relay 1
        {"id": 2, "rect": [...], "relay_id": 2}   ← Zone 2 → Relay 2
      ]
    },
    {
      "id": 2,
      "zones": [
        {"id": 3, "rect": [...], "relay_id": 3},  ← Zone 3 → Relay 3
        {"id": 4, "rect": [...], "relay_id": 4}   ← Zone 4 → Relay 4
      ]
    }
  ]
}

DETECTION FLOW:
1. DetectionWorker receives zones list: [(zone_id, rect, relay_id), ...]
2. When person detected, check each zone
3. If violation: on_violation(camera_id, zone_id, relay_id, frame)
4. Callback triggers: relay_manager.trigger(relay_id)
5. RelayManager activates the specific relay channel
"""


# =============================================================================
# HOW TO USE USB RELAY - SETUP INSTRUCTIONS
# =============================================================================
"""
STEP 1: Install pyhid library
    pip install pyhid

STEP 2: Find your USB relay's vendor/product ID
    import hid
    for device in hid.enumerate():
        print(f"VID: 0x{device['vendor_id']:04x}, PID: 0x{device['product_id']:04x}")
        print(f"  {device['manufacturer_string']} - {device['product_string']}")

STEP 3: Update config/app_settings.py:
    use_usb_relay = True
    usb_vendor_id = 0x16c0  # Your actual vendor ID
    usb_product_id = 0x05df  # Your actual product ID
    usb_num_channels = 4  # Number of channels on your board

STEP 4: Run application - it will automatically use USB relay

TROUBLESHOOTING:
- Linux: Add udev rules for permissions
- Windows: May need driver installation
- Check device is recognized: lsusb (Linux) or Device Manager (Windows)
"""


# =============================================================================
# PERFORMANCE TUNING GUIDE
# =============================================================================
"""
ALL TWEAKABLE PARAMETERS ARE IN: config/app_settings.py

SCENARIO 1: NEED MORE SPEED
    processing_resolution = (960, 540)  # Lower resolution
    yolo_model = "yolov8n.pt"  # Fastest model
    detection_confidence = 0.5
    ui_update_fps = 20  # Lower UI refresh

SCENARIO 2: NEED MORE ACCURACY
    processing_resolution = (1920, 1080)  # Higher resolution
    yolo_model = "yolov8m.pt"  # Better model
    detection_confidence = 0.4  # More sensitive
    ui_update_fps = 30

SCENARIO 3: SENSITIVE ZONES (trigger on any overlap)
    violation_mode = 'overlap'  # Trigger on any overlap
    detection_confidence = 0.45

SCENARIO 4: STRICT ZONES (only when person fully inside)
    violation_mode = 'center'  # Only when center inside
    detection_confidence = 0.6

TESTING DIFFERENT SETTINGS:
1. Edit config/app_settings.py
2. Restart application
3. Test detection behavior
4. Adjust as needed
"""


# =============================================================================
# HOW TO FIND YOUR USB RELAY - TESTING COMMANDS
# =============================================================================
"""
METHOD 1: Using pyhid-usb-relay command line tool
    
    # List all connected USB relay devices
    pyhid-usb-relay enum
    
    # Show current state of all relays
    pyhid-usb-relay state
    
    # Test relay 1
    pyhid-usb-relay on 1      # Turn on relay 1
    pyhid-usb-relay off 1     # Turn off relay 1
    pyhid-usb-relay toggle 1  # Toggle relay 1
    
    # Get device serial number
    pyhid-usb-relay get-serial


METHOD 2: Using Python script to test
    
    import pyhid_usb_relay
    
    # Find first relay device
    relay = pyhid_usb_relay.find()
    
    if relay:
        print(f"Relay found!")
        print(f"Current state: {relay.state}")
        
        # Test relay 1
        print("Testing relay 1...")
        relay.toggle_state(1)
        print(f"State after toggle: {relay.state}")
        
        # Turn it back off
        relay.toggle_state(1)
        print(f"State after second toggle: {relay.state}")
    else:
        print("No relay device found")


METHOD 3: Find relay with specific serial (if multiple devices)
    
    import pyhid_usb_relay
    
    # Enumerate all devices
    devices = pyhid_usb_relay.enumerate()
    for device in devices:
        print(f"Device: {device}")
    
    # Connect to specific device
    relay = pyhid_usb_relay.find(serial="ABCD1234")


INTERPRETING relay.state VALUE:
    
    The state is a byte where each bit represents a relay:
    - Bit 0 (LSB) = Relay 1
    - Bit 1 = Relay 2
    - Bit 2 = Relay 3
    - etc.
    
    Example:
    state = 1  (binary: 00000001) → Relay 1 is ON
    state = 3  (binary: 00000011) → Relay 1 and 2 are ON
    state = 5  (binary: 00000101) → Relay 1 and 3 are ON
    state = 0  (binary: 00000000) → All relays OFF
"""


# =============================================================================
# LINUX PERMISSIONS SETUP (if needed)
# =============================================================================
"""
If you get permission errors on Linux, create a udev rule:

1. Create file: /etc/udev/rules.d/99-usb-relay.rules

2. Add this content (adjust vendor/product ID if needed):
   SUBSYSTEM=="usb", ATTR{idVendor}=="16c0", ATTR{idProduct}=="05df", MODE="0666"

3. Reload udev rules:
   sudo udevadm control --reload-rules
   sudo udevadm trigger

4. Reconnect USB relay device

5. Test with:
   pyhid-usb-relay enum
"""