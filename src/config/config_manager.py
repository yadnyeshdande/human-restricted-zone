
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
