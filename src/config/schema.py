
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
