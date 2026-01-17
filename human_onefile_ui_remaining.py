# =============================================================================
# File: camera/camera_manager.py
# =============================================================================
"""Manage multiple camera workers."""

import queue
from typing import Dict, Optional, Tuple
import numpy as np
from utils.logger import get_logger
from .camera_worker import CameraWorker

logger = get_logger("CameraManager")


class CameraManager:
    """Manage lifecycle of multiple cameras."""
    
    def __init__(self, processing_resolution: Tuple[int, int] = (1280, 720)):
        """Initialize camera manager.
        
        Args:
            processing_resolution: Target resolution for all cameras
        """
        self.processing_resolution = processing_resolution
        self.workers: Dict[int, CameraWorker] = {}
        self.queues: Dict[int, queue.Queue] = {}
    
    def add_camera(self, camera_id: int, rtsp_url: str) -> bool:
        """Add and start a camera.
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP stream URL
        
        Returns:
            True if camera added successfully
        """
        if camera_id in self.workers:
            logger.warning(f"Camera {camera_id} already exists")
            return False
        
        try:
            output_queue = queue.Queue(maxsize=30)
            worker = CameraWorker(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                processing_resolution=self.processing_resolution,
                output_queue=output_queue
            )
            
            self.workers[camera_id] = worker
            self.queues[camera_id] = output_queue
            worker.start()
            
            logger.info(f"Camera {camera_id} added and started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add camera {camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove and stop a camera.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            True if camera removed successfully
        """
        if camera_id not in self.workers:
            return False
        
        worker = self.workers[camera_id]
        worker.stop()
        worker.join(timeout=5.0)
        
        del self.workers[camera_id]
        del self.queues[camera_id]
        
        logger.info(f"Camera {camera_id} removed")
        return True
    
    def get_frame_queue(self, camera_id: int) -> Optional[queue.Queue]:
        """Get frame queue for a camera."""
        return self.queues.get(camera_id)
    
    def get_latest_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Get latest frame from a camera."""
        worker = self.workers.get(camera_id)
        return worker.get_latest_frame() if worker else None
    
    def get_fps(self, camera_id: int) -> float:
        """Get FPS for a camera."""
        worker = self.workers.get(camera_id)
        return worker.get_fps() if worker else 0.0
    
    def is_connected(self, camera_id: int) -> bool:
        """Check if camera is connected."""
        worker = self.workers.get(camera_id)
        return worker.is_connected if worker else False
    
    def shutdown(self) -> None:
        """Stop all cameras."""
        logger.info("Shutting down all cameras")
        for camera_id in list(self.workers.keys()):
            self.remove_camera(camera_id)

# =============================================================================
# File: camera/camera_worker.py
# =============================================================================
"""RTSP camera capture worker."""

import cv2
import time
import queue
from typing import Optional, Tuple
import numpy as np
from utils.threading import StoppableThread
from utils.logger import get_logger
from utils.time_utils import FPSCounter
from .reconnect_policy import ReconnectPolicy

logger = get_logger("CameraWorker")


class CameraWorker(StoppableThread):
    """Capture frames from RTSP camera."""
    
    def __init__(
        self,
        camera_id: int,
        rtsp_url: str,
        processing_resolution: Tuple[int, int] = (1280, 720),
        output_queue: Optional[queue.Queue] = None
    ):
        """Initialize camera worker.
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP stream URL
            processing_resolution: Target resolution (width, height)
            output_queue: Queue to send captured frames
        """
        super().__init__(name=f"Camera-{camera_id}")
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.processing_resolution = processing_resolution
        self.output_queue = output_queue or queue.Queue(maxsize=30)
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.reconnect_policy = ReconnectPolicy()
        self.fps_counter = FPSCounter()
        self.current_fps = 0.0
        self.is_connected = False
        self.latest_frame: Optional[np.ndarray] = None
    
    def connect(self) -> bool:
        """Connect to RTSP stream."""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            if self.cap.isOpened():
                logger.info(f"Camera {self.camera_id} connected: {self.rtsp_url}")
                self.is_connected = True
                self.reconnect_policy.reset()
                return True
            else:
                logger.error(f"Camera {self.camera_id} failed to open")
                self.is_connected = False
                return False
                
        except Exception as e:
            logger.error(f"Camera {self.camera_id} connection error: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        logger.info(f"Camera {self.camera_id} disconnected")
    
    def run(self) -> None:
        """Main capture loop."""
        logger.info(f"Camera worker started for camera {self.camera_id}")
        
        while not self.stopped():
            # Connect if not connected
            if not self.is_connected:
                if not self.connect():
                    self.reconnect_policy.wait()
                    continue
            
            try:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Camera {self.camera_id} frame read failed")
                    self.disconnect()
                    continue
                
                # Resize to processing resolution
                frame = cv2.resize(frame, self.processing_resolution)
                
                # Store latest frame
                self.latest_frame = frame.copy()
                
                # Send to queue (non-blocking)
                try:
                    self.output_queue.put_nowait(frame)
                except queue.Full:
                    # Drop frame if queue is full
                    pass
                
                # Update FPS
                self.current_fps = self.fps_counter.update()
                
            except Exception as e:
                logger.error(f"Camera {self.camera_id} capture error: {e}")
                self.disconnect()
        
        self.disconnect()
        logger.info(f"Camera worker stopped for camera {self.camera_id}")
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get most recent frame."""
        return self.latest_frame

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
        backoff_factor: float = 2.0
    ):
        """Initialize reconnect policy.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for each retry
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = initial_delay
        self.attempt = 0
    
    def wait(self) -> None:
        """Wait before next reconnection attempt."""
        logger.info(f"Reconnect attempt {self.attempt + 1}, waiting {self.current_delay:.1f}s")
        time.sleep(self.current_delay)
        
        self.attempt += 1
        self.current_delay = min(
            self.current_delay * self.backoff_factor,
            self.max_delay
        )
    
    def reset(self) -> None:
        """Reset policy after successful connection."""
        self.current_delay = self.initial_delay
        self.attempt = 0

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
# File: config/config_manager.py
# =============================================================================
"""Configuration persistence and management."""

import json
from pathlib import Path
from typing import Optional, List, Tuple
from .schema import AppConfig, Camera, Zone
from utils.logger import get_logger
from utils.time_utils import get_timestamp


logger = get_logger("ConfigManager")


class ConfigManager:
    """Manage configuration persistence."""
    
    CONFIG_FILE = "human_boundaries.json"
    
    def __init__(self):
        self.config_path = Path(self.CONFIG_FILE)
        self.config: Optional[AppConfig] = None
        self._next_camera_id = 1
        self._next_zone_id = 1
        self._next_relay_id = 1
    
    def load(self) -> AppConfig:
        """Load configuration from file."""
        from config.app_settings import SETTINGS
        
        if not self.config_path.exists():
            logger.info("Config file not found, creating new configuration")
            self.config = AppConfig(processing_resolution=SETTINGS.processing_resolution)
            self.save()
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            self.config = AppConfig.from_dict(data)
            
            # Update ID counters
            if self.config.cameras:
                self._next_camera_id = max(c.id for c in self.config.cameras) + 1
                all_zones = [z for c in self.config.cameras for z in c.zones]
                if all_zones:
                    self._next_zone_id = max(z.id for z in all_zones) + 1
                    self._next_relay_id = max(z.relay_id for z in all_zones) + 1
            
            logger.info(f"Loaded configuration: {len(self.config.cameras)} cameras")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = AppConfig()
            return self.config
    
    def save(self) -> bool:
        """Save configuration to file."""
        if self.config is None:
            return False
        
        try:
            self.config.timestamp = get_timestamp()
            with open(self.config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def add_camera(self, rtsp_url: str) -> Camera:
        """Add a new camera."""
        camera = Camera(id=self._next_camera_id, rtsp_url=rtsp_url)
        self._next_camera_id += 1
        self.config.cameras.append(camera)
        logger.info(f"Added camera {camera.id}: {rtsp_url}")
        return camera
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera."""
        initial_len = len(self.config.cameras)
        self.config.cameras = [c for c in self.config.cameras if c.id != camera_id]
        removed = len(self.config.cameras) < initial_len
        if removed:
            logger.info(f"Removed camera {camera_id}")
        return removed
    
    def add_zone(self, camera_id: int, rect: Tuple[int, int, int, int]) -> Optional[Zone]:
        """Add a zone to a camera with automatic relay assignment."""
        camera = self.get_camera(camera_id)
        if camera is None:
            return None
        
        zone = Zone(
            id=self._next_zone_id,
            rect=rect,
            relay_id=self._next_relay_id
        )
        self._next_zone_id += 1
        self._next_relay_id += 1
        
        camera.zones.append(zone)
        logger.info(f"Added zone {zone.id} to camera {camera_id}, assigned relay {zone.relay_id}")
        return zone
    
    def remove_zone(self, camera_id: int, zone_id: int) -> bool:
        """Remove a zone."""
        camera = self.get_camera(camera_id)
        if camera is None:
            return False
        
        initial_len = len(camera.zones)
        camera.zones = [z for z in camera.zones if z.id != zone_id]
        removed = len(camera.zones) < initial_len
        if removed:
            logger.info(f"Removed zone {zone_id} from camera {camera_id}")
        return removed
    
    def update_zone(self, camera_id: int, zone_id: int, rect: Tuple[int, int, int, int]) -> bool:
        """Update a zone's rectangle."""
        camera = self.get_camera(camera_id)
        if camera is None:
            return False
        
        for zone in camera.zones:
            if zone.id == zone_id:
                zone.rect = rect
                logger.info(f"Updated zone {zone_id} in camera {camera_id}")
                return True
        return False
    
    def get_camera(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID."""
        for camera in self.config.cameras:
            if camera.id == camera_id:
                return camera
        return None
    
    def get_all_cameras(self) -> list:
        """Get all cameras."""
        return self.config.cameras if self.config else []

# =============================================================================
# File: config/migration.py
# =============================================================================
"""Configuration migration and backward compatibility."""

from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("Migration")


def migrate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old config formats to current version."""
    version = data.get('app_version', '0.0.0')
    
    if version == '1.0.0':
        return data
    
    # Add migration logic here for future versions
    logger.info(f"Migrated config from version {version} to 1.0.0")
    data['app_version'] = '1.0.0'
    return data


# =============================================================================
# File: config/schema.py
# =============================================================================
"""Data models and schemas."""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import json


@dataclass
class Zone:
    """Restricted zone definition."""
    id: int
    rect: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in processing resolution
    relay_id: int
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'rect': list(self.rect),
            'relay_id': self.relay_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Zone':
        return cls(
            id=data['id'],
            rect=tuple(data['rect']),
            relay_id=data['relay_id']
        )


@dataclass
class Camera:
    """Camera configuration."""
    id: int
    rtsp_url: str
    zones: List[Zone] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'rtsp_url': self.rtsp_url,
            'zones': [z.to_dict() for z in self.zones]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Camera':
        return cls(
            id=data['id'],
            rtsp_url=data['rtsp_url'],
            zones=[Zone.from_dict(z) for z in data.get('zones', [])]
        )


@dataclass
class AppConfig:
    """Application configuration."""
    app_version: str = "1.0.0"
    timestamp: str = ""
    processing_resolution: Tuple[int, int] = (1280, 720)
    cameras: List[Camera] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'app_version': self.app_version,
            'timestamp': self.timestamp,
            'processing_resolution': list(self.processing_resolution),
            'cameras': [c.to_dict() for c in self.cameras]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AppConfig':
        return cls(
            app_version=data.get('app_version', '1.0.0'),
            timestamp=data.get('timestamp', ''),
            processing_resolution=tuple(data.get('processing_resolution', [1280, 720])),
            cameras=[Camera.from_dict(c) for c in data.get('cameras', [])]
        )

# =============================================================================
# File: detection/detection_worker.py
# =============================================================================
"""Detection pipeline worker."""

import time
import queue
from typing import Optional, List, Tuple, Callable
import numpy as np
from utils.threading import StoppableThread
from utils.logger import get_logger
from utils.time_utils import FPSCounter
from .detector import PersonDetector
from .geometry import bbox_center, point_in_rect

logger = get_logger("DetectionWorker")


class DetectionWorker(StoppableThread):
    """Run detection pipeline for a camera."""
    
    def __init__(
        self,
        camera_id: int,
        frame_queue: queue.Queue,
        zones: List[Tuple[int, Tuple[int, int, int, int], int]],  # [(zone_id, rect, relay_id)]
        on_violation: Callable[[int, int, int, np.ndarray], None]  # (camera_id, zone_id, relay_id, frame)
    ):
        """Initialize detection worker.
        
        Args:
            camera_id: Camera identifier
            frame_queue: Queue to receive frames
            zones: List of (zone_id, rect, relay_id)
            on_violation: Callback when person enters restricted zone
        """
        super().__init__(name=f"Detection-{camera_id}")
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.zones = zones
        self.on_violation = on_violation
        self.fps_counter = FPSCounter()
        self.current_fps = 0.0
        
        try:
            self.detector = PersonDetector()
            logger.info(f"Detection worker initialized for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise
    
    def run(self) -> None:
        """Main detection loop."""
        logger.info(f"Detection worker started for camera {self.camera_id}")
        
        while not self.stopped():
            try:
                # Get frame with timeout
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Run detection
                persons = self.detector.detect_persons(frame)
                
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
                
                # Update FPS
                self.current_fps = self.fps_counter.update()
                
            except Exception as e:
                logger.error(f"Detection error for camera {self.camera_id}: {e}")
                time.sleep(0.1)
        
        logger.info(f"Detection worker stopped for camera {self.camera_id}")
    
    def get_fps(self) -> float:
        """Get current detection FPS."""
        return self.current_fps
    
    def update_zones(self, zones: List[Tuple[int, Tuple[int, int, int, int], int]]) -> None:
        """Update restricted zones."""
        self.zones = zones
        logger.info(f"Updated zones for camera {self.camera_id}: {len(zones)} zones")

# =============================================================================
# File: detection/detector.py
# =============================================================================
"""YOLO detector wrapper."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("Detector")


class PersonDetector:
    """YOLO-based person detector."""
    
    PERSON_CLASS_ID = 0
    
    def __init__(self, model_name: str = None, conf_threshold: float = None):
        from config.app_settings import SETTINGS
        
        self.conf_threshold = conf_threshold or SETTINGS.detection_confidence
        model_name = model_name or SETTINGS.yolo_model
        
        logger.info(f"Initializing YOLO detector: {model_name}, confidence: {self.conf_threshold}")
        """Initialize detector.
        
        Args:
            model_name: YOLO model name
            conf_threshold: Confidence threshold
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.device = 'cpu'
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            
            # Try to use CUDA if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.info("YOLO using GPU (CUDA)")
                else:
                    logger.info("YOLO using CPU")
            except ImportError:
                logger.info("YOLO using CPU (PyTorch not available)")
                
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons in frame.
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                classes=[self.PERSON_CLASS_ID],
                device=self.device,
                verbose=False
            )
            
            persons = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        persons.append((int(x1), int(y1), int(x2), int(y2)))
            
            return persons
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
        

# =============================================================================
# File: detection/geometry.py
# =============================================================================
"""Geometric calculations."""

from typing import Tuple


def point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    """Check if point is inside rectangle.
    
    Args:
        point: (x, y) coordinates
        rect: (x1, y1, x2, y2) rectangle
    
    Returns:
        True if point is inside rectangle
    """
    x, y = point
    x1, y1, x2, y2 = rect
    
    # Ensure proper ordering
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    return x_min <= x <= x_max and y_min <= y <= y_max


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate bounding box center.
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (cx, cy) center coordinates
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

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

# =============================================================================
# File: ui/detection_page.py
# =============================================================================
"""Detection mode interface for live monitoring."""

import cv2
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from relay.relay_manager import RelayManager
from detection.detection_worker import DetectionWorker
from .video_panel import VideoPanel
from utils.logger import get_logger

logger = get_logger("DetectionPage")


class DetectionPage(QWidget):
    """Live detection interface."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        camera_manager: CameraManager,
        relay_manager: RelayManager,
        parent=None
    ):
        """Initialize detection page.
        
        Args:
            config_manager: Configuration manager
            camera_manager: Camera manager
            relay_manager: Relay manager
            parent: Parent widget
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.camera_manager = camera_manager
        self.relay_manager = relay_manager
        
        # Detection state
        self.is_running = False
        self.detection_workers: Dict[int, DetectionWorker] = {}
        self.detection_queues: Dict[int, queue.Queue] = {}
        
        # Video panels
        self.video_panels: Dict[int, VideoPanel] = {}
        self.panel_containers: Dict[int, QWidget] = {}
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(33)  # ~30 FPS
        
        # Snapshot directory
        self.snapshot_dir = Path("snapshots")
        self.snapshot_dir.mkdir(exist_ok=True)
        
        self._setup_ui()
        self._load_cameras()
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._start_detection)
        toolbar.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self._stop_detection)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)
        
        toolbar.addStretch()
        
        self.status_label = QLabel("Detection Stopped")
        self.status_label.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self.status_label)
        
        layout.addLayout(toolbar)
        
        # Camera grid (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.camera_grid_widget = QWidget()
        self.camera_grid = QGridLayout(self.camera_grid_widget)
        self.camera_grid.setSpacing(10)
        scroll_area.setWidget(self.camera_grid_widget)
        
        layout.addWidget(scroll_area)
        
        # Status panel
        self.stats_label = QLabel("Waiting to start...")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.stats_label)
    
    def _load_cameras(self) -> None:
        """Load cameras and zones from configuration."""
        cameras = self.config_manager.get_all_cameras()
        
        for camera in cameras:
            self._add_camera_panel(camera.id, camera.rtsp_url)
    
    def _add_camera_panel(self, camera_id: int, rtsp_url: str) -> None:
        """Add camera panel to grid."""
        # Calculate grid position
        num_cameras = len(self.video_panels)
        cols = 2  # 2 columns
        row = num_cameras // cols
        col = num_cameras % cols
        
        # Create container
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create video panel
        video_panel = VideoPanel(
            camera_id=camera_id,
            processing_resolution=self.config_manager.config.processing_resolution
        )
        video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_panel.setMinimumSize(400, 300)
        
        container_layout.addWidget(video_panel)
        
        # Add to grid
        self.camera_grid.addWidget(container, row, col)
        
        # Store references
        self.video_panels[camera_id] = video_panel
        self.panel_containers[camera_id] = container
        
        # Load zones for display
        self._load_zones_for_camera(camera_id)
        
        logger.info(f"Detection panel {camera_id} added to UI")
    
    def _load_zones_for_camera(self, camera_id: int) -> None:
        """Load zones for a camera."""
        camera = self.config_manager.get_camera(camera_id)
        if not camera or camera_id not in self.video_panels:
            return
        
        zone_data = []
        for zone in camera.zones:
            color = self._get_zone_color(zone.relay_id)
            zone_data.append((
                zone.id,
                zone.rect,
                (color.red(), color.green(), color.blue())
            ))
        
        self.video_panels[camera_id].set_zones(zone_data)
    
    def _get_zone_color(self, relay_id: int) -> QColor:
        """Get color for relay ID."""
        colors = [
            QColor(0, 255, 0),      # Green
            QColor(0, 255, 255),    # Cyan
            QColor(255, 0, 255),    # Magenta
            QColor(255, 255, 0),    # Yellow
            QColor(255, 128, 0),    # Orange
            QColor(128, 0, 255),    # Purple
        ]
        return colors[(relay_id - 1) % len(colors)]
    
    def _start_detection(self) -> None:
        """Start detection on all cameras."""
        if self.is_running:
            return
        
        logger.info("Starting detection...")
        
        cameras = self.config_manager.get_all_cameras()
        
        for camera in cameras:
            # Get frame queue from camera manager
            frame_queue = self.camera_manager.get_frame_queue(camera.id)
            if not frame_queue:
                logger.warning(f"No frame queue for camera {camera.id}")
                continue
            
            # Prepare zones data
            zones_data = [
                (zone.id, zone.rect, zone.relay_id)
                for zone in camera.zones
            ]
            
            # Create detection worker
            worker = DetectionWorker(
                camera_id=camera.id,
                frame_queue=frame_queue,
                zones=zones_data,
                on_violation=self._handle_violation
            )
            
            self.detection_workers[camera.id] = worker
            worker.start()
            
            logger.info(f"Detection started for camera {camera.id}")
        
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Detection Running")
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
    
    def _stop_detection(self) -> None:
        """Stop detection on all cameras."""
        if not self.is_running:
            return
        
        logger.info("Stopping detection...")
        
        for camera_id, worker in self.detection_workers.items():
            worker.stop()
            worker.join(timeout=5.0)
            logger.info(f"Detection stopped for camera {camera_id}")
        
        self.detection_workers.clear()
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Detection Stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
    
    def _handle_violation(
        self,
        camera_id: int,
        zone_id: int,
        relay_id: int,
        frame: np.ndarray
    ) -> None:
        """Handle zone violation.
        
        Args:
            camera_id: Camera identifier
            zone_id: Zone identifier
            relay_id: Relay identifier
            frame: Frame with violation
        """
        logger.warning(
            f"VIOLATION DETECTED: Camera {camera_id}, Zone {zone_id}, Relay {relay_id}"
        )
        
        # Trigger relay
        triggered = self.relay_manager.trigger(relay_id)
        
        if triggered:
            logger.info(f"Relay {relay_id} triggered")
        else:
            logger.debug(f"Relay {relay_id} in cooldown")
        
        # Save snapshot
        self._save_snapshot(camera_id, zone_id, relay_id, frame)
    
    def _save_snapshot(
        self,
        camera_id: int,
        zone_id: int,
        relay_id: int,
        frame: np.ndarray
    ) -> None:
        """Save violation snapshot."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"violation_cam{camera_id}_zone{zone_id}_relay{relay_id}_{timestamp}.jpg"
            filepath = self.snapshot_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            logger.info(f"Snapshot saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def _update_displays(self) -> None:
        """Update video displays and statistics."""
        for camera_id, video_panel in self.video_panels.items():
            frame = self.camera_manager.get_latest_frame(camera_id)
            if frame is not None:
                video_panel.update_frame(frame)
                
                # Update info
                cap_fps = self.camera_manager.get_fps(camera_id)
                connected = self.camera_manager.is_connected(camera_id)
                
                det_fps = 0.0
                if camera_id in self.detection_workers:
                    det_fps = self.detection_workers[camera_id].get_fps()
                
                status = "Connected" if connected else "Disconnected"
                info = f"Camera {camera_id} | {status} | Cap: {cap_fps:.1f} FPS"
                if self.is_running:
                    info += f" | Det: {det_fps:.1f} FPS"
                
                video_panel.update_info(info)
        
        # Update statistics
        if self.is_running:
            total_zones = sum(
                len(cam.zones)
                for cam in self.config_manager.get_all_cameras()
            )
            self.stats_label.setText(
                f"Monitoring {len(self.video_panels)} cameras, "
                f"{total_zones} restricted zones"
            )
    
    def reload_configuration(self) -> None:
        """Reload configuration after teaching mode changes."""
        if self.is_running:
            self._stop_detection()
        
        # Reload zones
        for camera_id in self.video_panels.keys():
            self._load_zones_for_camera(camera_id)
        
        logger.info("Configuration reloaded")


# =============================================================================
# File: ui/main_window.py
# =============================================================================
"""Main application window."""

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QMessageBox, QAction,
    QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from relay.relay_manager import RelayManager
from .teaching_page import TeachingPage
from .detection_page import DetectionPage
from utils.logger import get_logger
from .settings_page import SettingsPage

logger = get_logger("MainWindow")


class MainWindow(QMainWindow):
    """Main application window with Teaching and Detection modes."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        camera_manager: CameraManager,
        relay_manager: RelayManager
    ):
        """Initialize main window.
        
        Args:
            config_manager: Configuration manager
            camera_manager: Camera manager
            relay_manager: Relay manager
        """
        super().__init__()
        self.config_manager = config_manager
        self.camera_manager = camera_manager
        self.relay_manager = relay_manager
        
        self.setWindowTitle("Industrial Vision Safety System")
        self.setMinimumSize(1280, 720)
        
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        # Central widget with tabs
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # Teaching page
        self.teaching_page = TeachingPage(
            self.config_manager,
            self.camera_manager
        )
        self.teaching_page.zones_changed.connect(self._on_zones_changed)
        self.tabs.addTab(self.teaching_page, "Teaching Mode")
        
        # Detection page
        self.detection_page = DetectionPage(
            self.config_manager,
            self.camera_manager,
            self.relay_manager
        )
        self.tabs.addTab(self.detection_page, "Detection Mode")
        
        # Settings page
        
        self.settings_page = SettingsPage(self.relay_manager)
        self.tabs.addTab(self.settings_page, "Settings")
        
        self.setCentralWidget(self.tabs)

    
    def _setup_menu(self) -> None:
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        save_action = QAction("Save Configuration", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.teaching_page._save_configuration)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        teaching_action = QAction("Teaching Mode", self)
        teaching_action.setShortcut("Ctrl+1")
        teaching_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(teaching_action)
        
        detection_action = QAction("Detection Mode", self)
        detection_action.setShortcut("Ctrl+2")
        detection_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(detection_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        # Shortcuts are already set in menu actions
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+3")
        settings_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        # view_menu.addAction(settings_action)
        pass
    
    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change."""
        if index == 1:  # Detection mode
            # Stop any running detection first
            if self.detection_page.is_running:
                self.detection_page._stop_detection()
            # Reload configuration
            self.detection_page.reload_configuration()
            logger.info("Switched to Detection Mode")
        else:  # Teaching mode
            logger.info("Switched to Teaching Mode")
    
    def _on_zones_changed(self) -> None:
        """Handle zones changed in teaching mode."""
        # Configuration is already updated by teaching page
        logger.debug("Zones configuration changed")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Vision Safety System",
            "<h2>Industrial Vision Safety System</h2>"
            "<p>Version 1.0.0</p>"
            "<p>Multi-camera vision safety monitoring with YOLO detection "
            "and relay-based alerting.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Multi-camera RTSP support</li>"
            "<li>Custom restricted zones</li>"
            "<li>Real-time person detection</li>"
            "<li>Automated relay triggering</li>"
            "<li>24/7 industrial operation</li>"
            "</ul>"
        )
    
    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Stop detection if running
        if self.detection_page.is_running:
            self.detection_page._stop_detection()
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?\n\nAll camera connections will be closed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            logger.info("Application closing...")
            
            # Save configuration
            self.config_manager.save()
            
            # Shutdown camera manager
            self.camera_manager.shutdown()
            
            event.accept()
        else:
            event.ignore()

# =============================================================================
# ADDITIONAL FILE: ui/settings_page.py (ADD THIS NEW FILE)
# =============================================================================
"""Settings page for application configuration."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QMessageBox, QLineEdit, QScrollArea
)
from PyQt5.QtCore import Qt
from config.app_settings import SETTINGS
from relay.relay_manager import RelayManager
from utils.logger import get_logger

logger = get_logger("SettingsPage")


class SettingsPage(QWidget):
    """Settings configuration interface."""
    
    def __init__(self, relay_manager: RelayManager, parent=None):
        """Initialize settings page.
        
        Args:
            relay_manager: Relay manager instance for testing
            parent: Parent widget
        """
        super().__init__(parent)
        self.relay_manager = relay_manager
        self.test_relay_index = 0
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        main_layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Application Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title)
        
        # Scrollable area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        settings_widget = QWidget()
        scroll_layout = QVBoxLayout(settings_widget)
        
        # Processing Settings Group
        processing_group = self._create_processing_group()
        scroll_layout.addWidget(processing_group)
        
        # Detection Settings Group
        detection_group = self._create_detection_group()
        scroll_layout.addWidget(detection_group)
        
        # Relay Settings Group
        relay_group = self._create_relay_group()
        scroll_layout.addWidget(relay_group)
        
        # Performance Settings Group
        performance_group = self._create_performance_group()
        scroll_layout.addWidget(performance_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(settings_widget)
        main_layout.addWidget(scroll)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self._save_settings)
        self.save_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        button_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        button_layout.addWidget(self.status_label)
        
        main_layout.addLayout(button_layout)
    
    def _create_processing_group(self) -> QGroupBox:
        """Create processing settings group."""
        group = QGroupBox("Processing Settings")
        layout = QFormLayout()
        
        # Resolution width
        self.resolution_width = QSpinBox()
        self.resolution_width.setRange(320, 3840)
        self.resolution_width.setSingleStep(160)
        self.resolution_width.setSuffix(" px")
        layout.addRow("Resolution Width:", self.resolution_width)
        
        # Resolution height
        self.resolution_height = QSpinBox()
        self.resolution_height.setRange(240, 2160)
        self.resolution_height.setSingleStep(90)
        self.resolution_height.setSuffix(" px")
        layout.addRow("Resolution Height:", self.resolution_height)
        
        # Quick presets
        preset_layout = QHBoxLayout()
        presets = [
            ("640×360", 640, 360),
            ("960×540", 960, 540),
            ("1280×720", 1280, 720),
            ("1920×1080", 1920, 1080)
        ]
        for name, w, h in presets:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, width=w, height=h: self._set_resolution(width, height))
            preset_layout.addWidget(btn)
        
        layout.addRow("Quick Presets:", preset_layout)
        
        # Info label
        info = QLabel("Lower resolution = Faster processing\nHigher resolution = Better accuracy")
        info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", info)
        
        group.setLayout(layout)
        return group
    
    def _create_detection_group(self) -> QGroupBox:
        """Create detection settings group."""
        group = QGroupBox("Detection Settings")
        layout = QFormLayout()
        
        # YOLO model selection
        self.yolo_model = QComboBox()
        models = [
            ("yolov8n.pt - Fastest, Lowest Accuracy", "yolov8n.pt"),
            ("yolov8s.pt - Fast, Good Accuracy", "yolov8s.pt"),
            ("yolov8m.pt - Medium Speed, Better Accuracy", "yolov8m.pt"),
            ("yolov8l.pt - Slow, High Accuracy", "yolov8l.pt"),
            ("yolov8x.pt - Slowest, Highest Accuracy", "yolov8x.pt")
        ]
        for display, value in models:
            self.yolo_model.addItem(display, value)
        layout.addRow("YOLO Model:", self.yolo_model)
        
        # Confidence threshold
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.1, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        layout.addRow("Confidence Threshold:", self.confidence)
        
        conf_info = QLabel("Lower = More detections (false positives)\nHigher = Fewer detections (may miss people)")
        conf_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", conf_info)
        
        # Violation mode
        self.violation_mode = QComboBox()
        self.violation_mode.addItem("Center - Only when person's center is inside zone", "center")
        self.violation_mode.addItem("Overlap - When person's box overlaps with zone", "overlap")
        layout.addRow("Violation Mode:", self.violation_mode)
        
        mode_info = QLabel(
            "Center = Stricter (person must be inside)\n"
            "Overlap = More sensitive (triggers on any overlap)"
        )
        mode_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", mode_info)
        
        group.setLayout(layout)
        return group
    
    def _create_relay_group(self) -> QGroupBox:
        """Create relay settings group."""
        group = QGroupBox("Relay Settings")
        layout = QFormLayout()
        
        # Use USB relay
        self.use_usb_relay = QCheckBox()
        self.use_usb_relay.stateChanged.connect(self._on_relay_type_changed)
        layout.addRow("Use USB Relay Hardware:", self.use_usb_relay)
        
        # Number of channels
        self.num_channels = QSpinBox()
        self.num_channels.setRange(1, 16)
        layout.addRow("Number of Channels:", self.num_channels)
        
        # Serial number (optional)
        self.usb_serial = QLineEdit()
        self.usb_serial.setPlaceholderText("Leave empty for auto-detect")
        layout.addRow("USB Serial (optional):", self.usb_serial)
        
        # Cooldown
        self.relay_cooldown = QDoubleSpinBox()
        self.relay_cooldown.setRange(0.5, 60.0)
        self.relay_cooldown.setSingleStep(0.5)
        self.relay_cooldown.setSuffix(" seconds")
        layout.addRow("Relay Cooldown:", self.relay_cooldown)
        
        # Duration
        self.relay_duration = QDoubleSpinBox()
        self.relay_duration.setRange(0.1, 10.0)
        self.relay_duration.setSingleStep(0.1)
        self.relay_duration.setSuffix(" seconds")
        layout.addRow("Relay Active Duration:", self.relay_duration)
        
        # Test relay button
        self.test_relay_btn = QPushButton("Test Relays")
        self.test_relay_btn.clicked.connect(self._start_relay_test)
        self.test_relay_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        layout.addRow("", self.test_relay_btn)
        
        relay_info = QLabel(
            "Test will toggle each relay one by one.\n"
            "Click 'Next' to proceed to the next relay."
        )
        relay_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", relay_info)
        
        group.setLayout(layout)
        return group
    
    def _create_performance_group(self) -> QGroupBox:
        """Create performance settings group."""
        group = QGroupBox("Performance Settings")
        layout = QFormLayout()
        
        # Frame queue size
        self.frame_queue_size = QSpinBox()
        self.frame_queue_size.setRange(5, 100)
        self.frame_queue_size.setSingleStep(5)
        layout.addRow("Frame Queue Size:", self.frame_queue_size)
        
        # UI update FPS
        self.ui_fps = QSpinBox()
        self.ui_fps.setRange(10, 60)
        self.ui_fps.setSingleStep(5)
        self.ui_fps.setSuffix(" FPS")
        layout.addRow("UI Update Rate:", self.ui_fps)
        
        perf_info = QLabel(
            "Larger queue = More memory, smoother playback\n"
            "Lower UI FPS = Less CPU usage"
        )
        perf_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", perf_info)
        
        group.setLayout(layout)
        return group
    
    def _set_resolution(self, width: int, height: int) -> None:
        """Set resolution preset."""
        self.resolution_width.setValue(width)
        self.resolution_height.setValue(height)
    
    def _on_relay_type_changed(self, state: int) -> None:
        """Handle relay type change."""
        if state == Qt.Checked:
            self.status_label.setText("USB relay enabled - restart required to apply")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.status_label.setText("Using relay simulator")
            self.status_label.setStyleSheet("color: blue;")
    
    def _load_settings(self) -> None:
        """Load current settings into UI."""
        # Processing
        self.resolution_width.setValue(SETTINGS.processing_resolution[0])
        self.resolution_height.setValue(SETTINGS.processing_resolution[1])
        
        # Detection
        index = self.yolo_model.findData(SETTINGS.yolo_model)
        if index >= 0:
            self.yolo_model.setCurrentIndex(index)
        
        self.confidence.setValue(SETTINGS.detection_confidence)
        
        index = self.violation_mode.findData(SETTINGS.violation_mode)
        if index >= 0:
            self.violation_mode.setCurrentIndex(index)
        
        # Relay
        self.use_usb_relay.setChecked(SETTINGS.use_usb_relay)
        self.num_channels.setValue(SETTINGS.usb_num_channels)
        if SETTINGS.usb_serial:
            self.usb_serial.setText(SETTINGS.usb_serial)
        self.relay_cooldown.setValue(SETTINGS.relay_cooldown)
        self.relay_duration.setValue(SETTINGS.relay_duration)
        
        # Performance
        self.frame_queue_size.setValue(SETTINGS.frame_queue_size)
        self.ui_fps.setValue(SETTINGS.ui_update_fps)
        
        logger.info("Settings loaded into UI")
    
    def _save_settings(self) -> None:
        """Save settings to file."""
        try:
            # Update SETTINGS object
            SETTINGS.processing_resolution = (
                self.resolution_width.value(),
                self.resolution_height.value()
            )
            SETTINGS.yolo_model = self.yolo_model.currentData()
            SETTINGS.detection_confidence = self.confidence.value()
            SETTINGS.violation_mode = self.violation_mode.currentData()
            
            SETTINGS.use_usb_relay = self.use_usb_relay.isChecked()
            SETTINGS.usb_num_channels = self.num_channels.value()
            serial_text = self.usb_serial.text().strip()
            SETTINGS.usb_serial = serial_text if serial_text else None
            SETTINGS.relay_cooldown = self.relay_cooldown.value()
            SETTINGS.relay_duration = self.relay_duration.value()
            
            SETTINGS.frame_queue_size = self.frame_queue_size.value()
            SETTINGS.ui_update_fps = self.ui_fps.value()
            
            # Save to file
            SETTINGS.save()
            
            self.status_label.setText("Settings saved successfully!")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            
            QMessageBox.information(
                self,
                "Settings Saved",
                "Settings saved successfully!\n\n"
                "Note: Some settings require application restart to take effect:\n"
                "- Processing resolution\n"
                "- YOLO model\n"
                "- USB relay configuration"
            )
            
            logger.info("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            self.status_label.setText("Error saving settings!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")
    
    def _reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to default values?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            SETTINGS.reset_to_defaults()
            self._load_settings()
            self.status_label.setText("Settings reset to defaults")
            self.status_label.setStyleSheet("color: blue;")
            logger.info("Settings reset to defaults")
    
    def _start_relay_test(self) -> None:
        """Start relay testing sequence."""
        self.test_relay_index = 1
        self._test_next_relay()
    
    def _test_next_relay(self) -> None:
        """Test next relay in sequence."""
        max_channels = self.num_channels.value()
        
        if self.test_relay_index > max_channels:
            QMessageBox.information(
                self,
                "Test Complete",
                f"All {max_channels} relays tested successfully!"
            )
            self.status_label.setText("Relay test completed")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            return
        
        # Show dialog for current relay
        msg = QMessageBox(self)
        msg.setWindowTitle(f"Testing Relay {self.test_relay_index}")
        msg.setText(
            f"Testing Relay {self.test_relay_index} of {max_channels}\n\n"
            f"The relay will now toggle ON then OFF.\n"
            f"Please observe the relay indicator/load.\n\n"
            f"Did Relay {self.test_relay_index} activate?"
        )
        msg.setIcon(QMessageBox.Question)
        
        yes_btn = msg.addButton("Yes, it worked", QMessageBox.YesRole)
        no_btn = msg.addButton("No, didn't work", QMessageBox.NoRole)
        next_btn = msg.addButton("Next Relay", QMessageBox.AcceptRole)
        cancel_btn = msg.addButton("Cancel Test", QMessageBox.RejectRole)
        
        # Activate relay
        try:
            logger.info(f"Testing relay {self.test_relay_index}")
            self.relay_manager.trigger(self.test_relay_index)
            self.status_label.setText(f"Testing Relay {self.test_relay_index}...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        except Exception as e:
            logger.error(f"Failed to test relay {self.test_relay_index}: {e}")
            QMessageBox.critical(
                self,
                "Relay Test Error",
                f"Failed to activate relay {self.test_relay_index}:\n{e}"
            )
            return
        
        msg.exec_()
        
        clicked = msg.clickedButton()
        
        if clicked == cancel_btn:
            self.status_label.setText("Relay test cancelled")
            self.status_label.setStyleSheet("color: gray;")
            return
        elif clicked == no_btn:
            QMessageBox.warning(
                self,
                "Relay Issue",
                f"Relay {self.test_relay_index} did not activate.\n\n"
                f"Possible issues:\n"
                f"- Relay hardware not connected\n"
                f"- Incorrect channel number\n"
                f"- Hardware failure\n"
                f"- Check logs for details"
            )
        elif clicked == yes_btn:
            logger.info(f"Relay {self.test_relay_index} test successful")
        
        # Move to next relay
        self.test_relay_index += 1
        self._test_next_relay()


# =============================================================================
# File: ui/teaching_page.py
# =============================================================================
"""Teaching mode interface for zone editing."""

import queue
from typing import Dict, List, Tuple, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QInputDialog, QMessageBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from .video_panel import VideoPanel
from .zone_editor import ZoneEditor
from utils.logger import get_logger

logger = get_logger("TeachingPage")


class TeachingPage(QWidget):
    """Zone editor interface with multi-camera support."""
    
    zones_changed = pyqtSignal()
    
    def __init__(
        self,
        config_manager: ConfigManager,
        camera_manager: CameraManager,
        parent=None
    ):
        """Initialize teaching page.
        
        Args:
            config_manager: Configuration manager
            camera_manager: Camera manager
            parent: Parent widget
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.camera_manager = camera_manager
        
        # Video panels and editors
        self.video_panels: Dict[int, VideoPanel] = {}
        self.zone_editors: Dict[int, ZoneEditor] = {}
        self.panel_containers: Dict[int, QWidget] = {}
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_frames)
        self.update_timer.start(33)  # ~30 FPS
        
        self._setup_ui()
        self._load_cameras()
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        
        self.add_camera_btn = QPushButton("Add Camera")
        self.add_camera_btn.clicked.connect(self._add_camera)
        toolbar.addWidget(self.add_camera_btn)
        
        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.clicked.connect(self._save_configuration)
        toolbar.addWidget(self.save_btn)
        
        self.clear_zones_btn = QPushButton("Clear All Zones")
        self.clear_zones_btn.clicked.connect(self._clear_all_zones)
        toolbar.addWidget(self.clear_zones_btn)
        
        toolbar.addStretch()
        
        self.status_label = QLabel("Ready")
        toolbar.addWidget(self.status_label)
        
        layout.addLayout(toolbar)
        
        # Camera grid (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.camera_grid_widget = QWidget()
        self.camera_grid = QGridLayout(self.camera_grid_widget)
        self.camera_grid.setSpacing(10)
        scroll_area.setWidget(self.camera_grid_widget)
        
        layout.addWidget(scroll_area)
        
        # Instructions
        instructions = QLabel(
            "Draw zones by clicking and dragging. "
            "Select zones to move or resize. "
            "Press Delete to remove selected zone. "
            "Ctrl+Z to undo, Ctrl+Y to redo."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(instructions)
    
    def _load_cameras(self) -> None:
        """Load cameras from configuration."""
        cameras = self.config_manager.get_all_cameras()
        for camera in cameras:
            self._add_camera_panel(camera.id, camera.rtsp_url)
            
            # Load zones
            for zone in camera.zones:
                self._add_zone_visual(camera.id, zone.id, zone.rect, zone.relay_id)
    
    def _add_camera(self) -> None:
        """Add a new camera."""
        rtsp_url, ok = QInputDialog.getText(
            self,
            "Add Camera",
            "Enter RTSP URL:\n(e.g., rtsp://admin:Pass_123@192.168.1.64:554/stream)",
            text="rtsp://"
        )
        
        if not ok or not rtsp_url:
            return
        
        # Add to configuration
        camera = self.config_manager.add_camera(rtsp_url)
        
        # Start camera capture
        if self.camera_manager.add_camera(camera.id, rtsp_url):
            self._add_camera_panel(camera.id, rtsp_url)
            self.status_label.setText(f"Camera {camera.id} added")
            logger.info(f"Camera {camera.id} added: {rtsp_url}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to connect to camera")
    
    def _add_camera_panel(self, camera_id: int, rtsp_url: str) -> None:
        """Add camera panel to grid.
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP URL
        """
        # Calculate grid position
        num_cameras = len(self.video_panels)
        cols = 2  # 2 columns
        row = num_cameras // cols
        col = num_cameras % cols
        
        # Create container
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create video panel
        video_panel = VideoPanel(
            camera_id=camera_id,
            processing_resolution=self.config_manager.config.processing_resolution
        )
        video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_panel.setMinimumSize(400, 300)
        
        # Create zone editor overlay
        zone_editor = ZoneEditor(video_panel)
        zone_editor.setGeometry(video_panel.video_label.geometry())
        zone_editor.zone_created.connect(
            lambda rect, cid=camera_id: self._on_zone_created(cid, rect)
        )
        zone_editor.zone_modified.connect(
            lambda zid, rect, cid=camera_id: self._on_zone_modified(cid, zid, rect)
        )
        
        container_layout.addWidget(video_panel)
        
        # Camera controls
        controls = QHBoxLayout()
        
        remove_btn = QPushButton(f"Remove Camera {camera_id}")
        remove_btn.clicked.connect(lambda: self._remove_camera(camera_id))
        controls.addWidget(remove_btn)
        
        delete_zone_btn = QPushButton("Delete Selected Zone")
        delete_zone_btn.clicked.connect(lambda: self._delete_selected_zone(camera_id))
        controls.addWidget(delete_zone_btn)
        
        controls.addStretch()
        container_layout.addLayout(controls)
        
        # Add to grid
        self.camera_grid.addWidget(container, row, col)
        
        # Store references
        self.video_panels[camera_id] = video_panel
        self.zone_editors[camera_id] = zone_editor
        self.panel_containers[camera_id] = container
        
        logger.info(f"Camera panel {camera_id} added to UI")
    
    def _remove_camera(self, camera_id: int) -> None:
        """Remove camera."""
        reply = QMessageBox.question(
            self,
            "Remove Camera",
            f"Remove camera {camera_id} and all its zones?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove from managers
            self.camera_manager.remove_camera(camera_id)
            self.config_manager.remove_camera(camera_id)
            
            # Remove from UI
            if camera_id in self.panel_containers:
                container = self.panel_containers[camera_id]
                self.camera_grid.removeWidget(container)
                container.deleteLater()
                
                del self.video_panels[camera_id]
                del self.zone_editors[camera_id]
                del self.panel_containers[camera_id]
            
            self._reorganize_grid()
            self.status_label.setText(f"Camera {camera_id} removed")
            self.zones_changed.emit()
    
    def _reorganize_grid(self) -> None:
        """Reorganize camera grid after removal."""
        # Remove all widgets
        for camera_id, container in self.panel_containers.items():
            self.camera_grid.removeWidget(container)
        
        # Re-add in order
        cols = 2
        for idx, (camera_id, container) in enumerate(sorted(self.panel_containers.items())):
            row = idx // cols
            col = idx % cols
            self.camera_grid.addWidget(container, row, col)
    
    def _on_zone_created(self, camera_id: int, rect: Tuple[int, int, int, int]) -> None:
        """Handle zone creation."""
        # Add to configuration
        zone = self.config_manager.add_zone(camera_id, rect)
        
        if zone:
            self._add_zone_visual(camera_id, zone.id, rect, zone.relay_id)
            self.status_label.setText(
                f"Zone {zone.id} created for Camera {camera_id}, "
                f"assigned to Relay {zone.relay_id}"
            )
            self.zones_changed.emit()
            logger.info(
                f"Zone created: Camera {camera_id}, Zone {zone.id}, "
                f"Relay {zone.relay_id}, Rect {rect}"
            )
    
    def _on_zone_modified(self, camera_id: int, zone_id: int, rect: Tuple[int, int, int, int]) -> None:
        """Handle zone modification."""
        if self.config_manager.update_zone(camera_id, zone_id, rect):
            self._update_zone_visuals(camera_id)
            self.status_label.setText(f"Zone {zone_id} updated")
            self.zones_changed.emit()
    
    def _add_zone_visual(
        self,
        camera_id: int,
        zone_id: int,
        rect: Tuple[int, int, int, int],
        relay_id: int
    ) -> None:
        """Add zone to visual editor."""
        if camera_id in self.zone_editors:
            color = self._get_zone_color(relay_id)
            self.zone_editors[camera_id].add_zone(zone_id, rect, color)
            self._update_zone_visuals(camera_id)
    
    def _update_zone_visuals(self, camera_id: int) -> None:
        """Update zone visuals on video panel."""
        if camera_id not in self.video_panels or camera_id not in self.zone_editors:
            return
        
        zones = self.zone_editors[camera_id].get_zones()
        
        # Get relay IDs from config
        camera = self.config_manager.get_camera(camera_id)
        if not camera:
            return
        
        zone_data = []
        for zone_id, rect in zones:
            # Find relay_id
            relay_id = None
            for zone in camera.zones:
                if zone.id == zone_id:
                    relay_id = zone.relay_id
                    break
            
            if relay_id:
                color = self._get_zone_color(relay_id)
                zone_data.append((zone_id, rect, (color.red(), color.green(), color.blue())))
        
        self.video_panels[camera_id].set_zones(zone_data)
    
    def _get_zone_color(self, relay_id: int) -> QColor:
        """Get color for relay ID."""
        colors = [
            QColor(0, 255, 0),      # Green
            QColor(0, 255, 255),    # Cyan
            QColor(255, 0, 255),    # Magenta
            QColor(255, 255, 0),    # Yellow
            QColor(255, 128, 0),    # Orange
            QColor(128, 0, 255),    # Purple
        ]
        return colors[(relay_id - 1) % len(colors)]
    
    def _delete_selected_zone(self, camera_id: int) -> None:
        """Delete selected zone for camera."""
        if camera_id not in self.zone_editors:
            return
        
        zone_id = self.zone_editors[camera_id].selected_zone_id
        if zone_id is not None:
            self.config_manager.remove_zone(camera_id, zone_id)
            self.zone_editors[camera_id].remove_zone(zone_id)
            self._update_zone_visuals(camera_id)
            self.status_label.setText(f"Zone {zone_id} deleted")
            self.zones_changed.emit()
    
    def _clear_all_zones(self) -> None:
        """Clear all zones from all cameras."""
        reply = QMessageBox.question(
            self,
            "Clear All Zones",
            "Remove all zones from all cameras?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for camera in self.config_manager.get_all_cameras():
                camera.zones.clear()
            
            for zone_editor in self.zone_editors.values():
                zone_editor.clear_zones()
            
            for camera_id in self.video_panels.keys():
                self._update_zone_visuals(camera_id)
            
            self.status_label.setText("All zones cleared")
            self.zones_changed.emit()
    
    def _save_configuration(self) -> None:
        """Save configuration to file."""
        if self.config_manager.save():
            self.status_label.setText("Configuration saved")
            QMessageBox.information(self, "Success", "Configuration saved successfully")
        else:
            QMessageBox.warning(self, "Error", "Failed to save configuration")
    
    def _update_frames(self) -> None:
        """Update video frames."""
        for camera_id, video_panel in self.video_panels.items():
            frame = self.camera_manager.get_latest_frame(camera_id)
            if frame is not None:
                video_panel.update_frame(frame)
                
                # Update info
                fps = self.camera_manager.get_fps(camera_id)
                connected = self.camera_manager.is_connected(camera_id)
                status = "Connected" if connected else "Disconnected"
                video_panel.update_info(f"Camera {camera_id} | {status} | {fps:.1f} FPS")
    
    def resizeEvent(self, event):
        """Handle resize to update zone editor geometry."""
        super().resizeEvent(event)
        for camera_id, video_panel in self.video_panels.items():
            if camera_id in self.zone_editors:
                zone_editor = self.zone_editors[camera_id]
                zone_editor.setGeometry(video_panel.video_label.geometry())



# =============================================================================
# File: ui/video_panel.py
# =============================================================================
"""Video panel widget with aspect ratio preservation."""

import cv2
import numpy as np
from typing import Optional, Tuple
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from utils.logger import get_logger

logger = get_logger("VideoPanel")


class VideoPanel(QWidget):
    """Display widget for camera feed with aspect ratio preservation."""
    
    # Signals
    frame_clicked = pyqtSignal(int, int)  # x, y in widget coordinates
    
    def __init__(
        self,
        camera_id: int,
        processing_resolution: Tuple[int, int] = (1280, 720),
        parent=None
    ):
        """Initialize video panel.
        
        Args:
            camera_id: Camera identifier
            processing_resolution: Processing resolution (width, height)
            parent: Parent widget
        """
        super().__init__(parent)
        self.camera_id = camera_id
        self.processing_resolution = processing_resolution
        self.processing_width, self.processing_height = processing_resolution
        
        # Frame data
        self.current_frame: Optional[np.ndarray] = None
        self.display_pixmap: Optional[QPixmap] = None
        
        # Letterbox/pillarbox offsets for coordinate mapping
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0
        
        # Zones to draw (list of (zone_id, rect, color))
        self.zones = []
        
        # UI setup
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #444;")
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setScaledContents(False)
        layout.addWidget(self.video_label)
        
        # Info label
        self.info_label = QLabel(f"Camera {self.camera_id}")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #ffffff; font-size: 10px; padding: 2px;")
        layout.addWidget(self.info_label)
    
    def update_frame(self, frame: np.ndarray) -> None:
        """Update displayed frame.
        
        Args:
            frame: BGR frame from OpenCV
        """
        self.current_frame = frame
        self._render_frame()
    
    def _render_frame(self) -> None:
        """Render frame with zones overlay."""
        if self.current_frame is None:
            return
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        # Draw zones on frame
        for zone_id, rect, color in self.zones:
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_rgb,
                f"Zone {zone_id}",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Convert to QPixmap with aspect ratio preservation
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget while preserving aspect ratio
        widget_size = self.video_label.size()
        pixmap = QPixmap.fromImage(q_image)
        
        # Calculate scaling to fit widget
        scaled_pixmap = pixmap.scaled(
            widget_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Calculate offsets for letterbox/pillarbox
        self.offset_x = (widget_size.width() - scaled_pixmap.width()) // 2
        self.offset_y = (widget_size.height() - scaled_pixmap.height()) // 2
        self.scale = scaled_pixmap.width() / width
        
        self.display_pixmap = scaled_pixmap
        self.video_label.setPixmap(scaled_pixmap)
    
    def set_zones(self, zones: list) -> None:
        """Set zones to display.
        
        Args:
            zones: List of (zone_id, rect, color) tuples
        """
        self.zones = zones
        if self.current_frame is not None:
            self._render_frame()
    
    def widget_to_processing(self, widget_x: int, widget_y: int) -> Tuple[int, int]:
        """Convert widget coordinates to processing coordinates.
        
        Args:
            widget_x: X coordinate in widget space
            widget_y: Y coordinate in widget space
        
        Returns:
            (x, y) in processing space
        """
        # Remove offset
        x = widget_x - self.offset_x
        y = widget_y - self.offset_y
        
        # Scale to processing resolution
        if self.scale > 0:
            x = int(x / self.scale)
            y = int(y / self.scale)
        
        # Clamp to processing resolution
        x = max(0, min(x, self.processing_width - 1))
        y = max(0, min(y, self.processing_height - 1))
        
        return x, y
    
    def processing_to_widget(self, proc_x: int, proc_y: int) -> Tuple[int, int]:
        """Convert processing coordinates to widget coordinates.
        
        Args:
            proc_x: X coordinate in processing space
            proc_y: Y coordinate in processing space
        
        Returns:
            (x, y) in widget space
        """
        # Scale from processing resolution
        x = int(proc_x * self.scale) + self.offset_x
        y = int(proc_y * self.scale) + self.offset_y
        
        return x, y
    
    def update_info(self, info_text: str) -> None:
        """Update info label text."""
        self.info_label.setText(info_text)
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            # Get position relative to video_label
            pos = self.video_label.mapFrom(self, event.pos())
            self.frame_clicked.emit(pos.x(), pos.y())


# =============================================================================
# File: ui/zone_editor.py
# =============================================================================
"""Zone drawing and editing widget."""

from typing import Optional, List, Tuple
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPaintEvent, QMouseEvent
from utils.logger import get_logger

logger = get_logger("ZoneEditor")


class ZoneEditor(QWidget):
    """Interactive zone drawing and editing overlay."""
    
    # Signals
    zone_created = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    zone_modified = pyqtSignal(int, tuple)  # zone_id, (x1, y1, x2, y2)
    zone_selected = pyqtSignal(int)  # zone_id
    
    # States
    STATE_IDLE = 0
    STATE_DRAWING = 1
    STATE_MOVING = 2
    STATE_RESIZING = 3
    
    def __init__(self, parent=None):
        """Initialize zone editor."""
        super().__init__(parent)
        
        # State
        self.state = self.STATE_IDLE
        self.zones: List[Tuple[int, QRect, QColor]] = []  # (id, rect, color)
        self.selected_zone_id: Optional[int] = None
        self.resize_handle: Optional[str] = None  # 'tl', 'tr', 'bl', 'br'
        
        # Drawing
        self.draw_start: Optional[QPoint] = None
        self.draw_current: Optional[QPoint] = None
        
        # Moving
        self.move_offset: Optional[QPoint] = None
        
        # Undo/Redo
        self.history: List[List[Tuple[int, QRect, QColor]]] = []
        self.history_index = -1
        
        # Configuration
        self.handle_size = 8
        self.default_color = QColor(0, 255, 0)
        self.selected_color = QColor(255, 255, 0)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        # Make background transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
    
    def set_zones(self, zones: List[Tuple[int, Tuple[int, int, int, int], QColor]]) -> None:
        """Set zones to display and edit.
        
        Args:
            zones: List of (zone_id, (x1, y1, x2, y2), color)
        """
        self.zones = [
            (zone_id, QRect(x1, y1, x2 - x1, y2 - y1), color)
            for zone_id, (x1, y1, x2, y2), color in zones
        ]
        self._save_history()
        self.update()
    
    def add_zone(self, zone_id: int, rect: Tuple[int, int, int, int], color: QColor = None) -> None:
        """Add a zone.
        
        Args:
            zone_id: Zone identifier
            rect: (x1, y1, x2, y2)
            color: Zone color
        """
        x1, y1, x2, y2 = rect
        self.zones.append((
            zone_id,
            QRect(x1, y1, x2 - x1, y2 - y1),
            color or self.default_color
        ))
        self._save_history()
        self.update()
    
    def remove_zone(self, zone_id: int) -> None:
        """Remove a zone."""
        self.zones = [z for z in self.zones if z[0] != zone_id]
        if self.selected_zone_id == zone_id:
            self.selected_zone_id = None
        self._save_history()
        self.update()
    
    def clear_zones(self) -> None:
        """Clear all zones."""
        self.zones.clear()
        self.selected_zone_id = None
        self._save_history()
        self.update()
    
    def get_zones(self) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """Get all zones.
        
        Returns:
            List of (zone_id, (x1, y1, x2, y2))
        """
        return [
            (zone_id, (rect.x(), rect.y(), rect.right(), rect.bottom()))
            for zone_id, rect, _ in self.zones
        ]
    
    def _save_history(self) -> None:
        """Save current state to history."""
        # Remove any redo history
        self.history = self.history[:self.history_index + 1]
        
        # Save current state
        self.history.append([
            (zid, QRect(rect), QColor(color))
            for zid, rect, color in self.zones
        ])
        self.history_index = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self) -> bool:
        """Undo last operation."""
        if self.history_index > 0:
            self.history_index -= 1
            self.zones = [
                (zid, QRect(rect), QColor(color))
                for zid, rect, color in self.history[self.history_index]
            ]
            self.update()
            return True
        return False
    
    def redo(self) -> bool:
        """Redo last undone operation."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.zones = [
                (zid, QRect(rect), QColor(color))
                for zid, rect, color in self.history[self.history_index]
            ]
            self.update()
            return True
        return False
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint zones and handles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw existing zones
        for zone_id, rect, color in self.zones:
            is_selected = (zone_id == self.selected_zone_id)
            
            # Draw rectangle
            pen_width = 3 if is_selected else 2
            pen_color = self.selected_color if is_selected else color
            painter.setPen(QPen(pen_color, pen_width))
            painter.setBrush(QBrush(QColor(pen_color.red(), pen_color.green(), pen_color.blue(), 30)))
            painter.drawRect(rect)
            
            # Draw zone label
            painter.setPen(QPen(pen_color, 1))
            painter.drawText(rect.x() + 5, rect.y() + 15, f"Zone {zone_id}")
            
            # Draw resize handles for selected zone
            if is_selected:
                self._draw_handles(painter, rect)
        
        # Draw current drawing
        if self.state == self.STATE_DRAWING and self.draw_start and self.draw_current:
            painter.setPen(QPen(self.default_color, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            rect = QRect(self.draw_start, self.draw_current).normalized()
            painter.drawRect(rect)
    
    def _draw_handles(self, painter: QPainter, rect: QRect) -> None:
        """Draw resize handles."""
        handle_color = self.selected_color
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(Qt.white, 1))
        
        handles = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]
        
        for handle_pos in handles:
            handle_rect = QRect(
                handle_pos.x() - self.handle_size // 2,
                handle_pos.y() - self.handle_size // 2,
                self.handle_size,
                self.handle_size
            )
            painter.drawRect(handle_rect)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if event.button() != Qt.LeftButton:
            return
        
        pos = event.pos()
        
        # Check if clicking on resize handle
        if self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect:
                handle = self._get_handle_at(pos, selected_rect)
                if handle:
                    self.state = self.STATE_RESIZING
                    self.resize_handle = handle
                    return
        
        # Check if clicking on existing zone
        clicked_zone = self._get_zone_at(pos)
        if clicked_zone is not None:
            self.selected_zone_id = clicked_zone
            self.zone_selected.emit(clicked_zone)
            self.state = self.STATE_MOVING
            selected_rect = self._get_zone_rect(clicked_zone)
            if selected_rect:
                self.move_offset = pos - selected_rect.topLeft()
            self.update()
            return
        
        # Start drawing new zone
        self.selected_zone_id = None
        self.state = self.STATE_DRAWING
        self.draw_start = pos
        self.draw_current = pos
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        pos = event.pos()
        
        if self.state == self.STATE_DRAWING:
            self.draw_current = pos
            self.update()
        
        elif self.state == self.STATE_MOVING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect and self.move_offset:
                new_pos = pos - self.move_offset
                selected_rect.moveTo(new_pos)
                self.update()
        
        elif self.state == self.STATE_RESIZING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect and self.resize_handle:
                self._resize_rect(selected_rect, pos, self.resize_handle)
                self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() != Qt.LeftButton:
            return
        
        if self.state == self.STATE_DRAWING:
            if self.draw_start and self.draw_current:
                rect = QRect(self.draw_start, self.draw_current).normalized()
                if rect.width() > 10 and rect.height() > 10:
                    self.zone_created.emit((
                        rect.x(), rect.y(), rect.right(), rect.bottom()
                    ))
            self.draw_start = None
            self.draw_current = None
        
        elif self.state == self.STATE_MOVING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect:
                self.zone_modified.emit(
                    self.selected_zone_id,
                    (selected_rect.x(), selected_rect.y(), selected_rect.right(), selected_rect.bottom())
                )
                self._save_history()
        
        elif self.state == self.STATE_RESIZING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect:
                self.zone_modified.emit(
                    self.selected_zone_id,
                    (selected_rect.x(), selected_rect.y(), selected_rect.right(), selected_rect.bottom())
                )
                self._save_history()
        
        self.state = self.STATE_IDLE
        self.update()
    
    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() == Qt.Key_Delete and self.selected_zone_id is not None:
            self.remove_zone(self.selected_zone_id)
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self.redo()
    
    def _get_zone_at(self, pos: QPoint) -> Optional[int]:
        """Get zone ID at position."""
        for zone_id, rect, _ in reversed(self.zones):
            if rect.contains(pos):
                return zone_id
        return None
    
    def _get_zone_rect(self, zone_id: int) -> Optional[QRect]:
        """Get zone rectangle by ID."""
        for zid, rect, _ in self.zones:
            if zid == zone_id:
                return rect
        return None
    
    def _get_handle_at(self, pos: QPoint, rect: QRect) -> Optional[str]:
        """Get resize handle at position."""
        handles = {
            'tl': rect.topLeft(),
            'tr': rect.topRight(),
            'bl': rect.bottomLeft(),
            'br': rect.bottomRight()
        }
        
        for handle_name, handle_pos in handles.items():
            handle_rect = QRect(
                handle_pos.x() - self.handle_size,
                handle_pos.y() - self.handle_size,
                self.handle_size * 2,
                self.handle_size * 2
            )
            if handle_rect.contains(pos):
                return handle_name
        return None
    
    def _resize_rect(self, rect: QRect, pos: QPoint, handle: str) -> None:
        """Resize rectangle by dragging handle."""
        if handle == 'tl':
            rect.setTopLeft(pos)
        elif handle == 'tr':
            rect.setTopRight(pos)
        elif handle == 'bl':
            rect.setBottomLeft(pos)
        elif handle == 'br':
            rect.setBottomRight(pos)
        
        # Ensure minimum size
        if rect.width() < 10:
            rect.setWidth(10)
        if rect.height() < 10:
            rect.setHeight(10)


# =============================================================================
# File: utils/logger.py
# =============================================================================
"""Centralized logging with rotating file handler."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class SystemLogger:
    """Thread-safe centralized logging system."""
    
    _instance: Optional['SystemLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not SystemLogger._initialized:
            self._setup_logging()
            SystemLogger._initialized = True
    
    def _setup_logging(self) -> None:
        """Configure logging with rotating file handler."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Root logger
        self.logger = logging.getLogger("VisionSafety")
        self.logger.setLevel(logging.DEBUG)
        
        # Rotating file handler (10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "vision_safety.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a child logger."""
        return self.logger.getChild(name)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return SystemLogger().get_logger(name)


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

# =============================================================================
# File: app.py - APPLICATION ENTRY POINT
# =============================================================================

import sys
import os

# Add src to path if running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from utils.logger import get_logger
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from relay.relay_manager import RelayManager

logger = get_logger("Main")


def main():
    """Application entry point."""
    logger.info("=" * 80)
    logger.info("INDUSTRIAL VISION SAFETY SYSTEM - STARTING")
    logger.info("=" * 80)
    
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Vision Safety System")
        app.setStyle("Fusion")  # Modern style
        
        # Initialize managers
        logger.info("Initializing configuration manager...")
        config_manager = ConfigManager()
        config = config_manager.load()
        
        logger.info("Initializing camera manager...")
        camera_manager = CameraManager(
            processing_resolution=config.processing_resolution
        )
        
        logger.info("Initializing relay manager...")
        from config.app_settings import SETTINGS
        
        # Choose relay interface based on settings
        relay_interface = None
        if SETTINGS.use_usb_relay:
            try:
                from relay.relay_usb_hid import RelayUSBHID
                relay_interface = RelayUSBHID(
                    num_channels=SETTINGS.usb_num_channels,
                    serial=SETTINGS.usb_serial
                )
                logger.info("Using pyhid_usb_relay hardware")
            except Exception as e:
                logger.error(f"Failed to initialize USB relay: {e}")
                logger.error("Falling back to relay simulator")
        
        relay_manager = RelayManager(
            interface=relay_interface,
            cooldown=SETTINGS.relay_cooldown,
            activation_duration=SETTINGS.relay_duration
        )
        
        # Start cameras from configuration
        for camera in config.cameras:
            logger.info(f"Starting camera {camera.id}: {camera.rtsp_url}")
            camera_manager.add_camera(camera.id, camera.rtsp_url)
        
        # Create and show main window
        logger.info("Creating main window...")
        window = MainWindow(config_manager, camera_manager, relay_manager)
        window.show()
        
        logger.info("Application initialized successfully")
        logger.info("=" * 80)
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()