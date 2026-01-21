
# =============================================================================
# File: config/schema.py
# =============================================================================
"""Data models and schemas."""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
import json


@dataclass
class Zone:
    """Restricted zone definition - supports polygon shapes."""
    id: int
    points: List[Tuple[int, int]]  # List of (x, y) points forming polygon
    relay_id: int
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'points': [list(p) for p in self.points],  # Convert tuples to lists for JSON
            'relay_id': self.relay_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Zone':
        # Support both old format (rect) and new format (points)
        if 'points' in data:
            return cls(
                id=data['id'],
                points=[tuple(p) for p in data['points']],
                relay_id=data['relay_id']
            )
        elif 'rect' in data:
            # Convert old rectangle format to polygon
            x1, y1, x2, y2 = data['rect']
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            return cls(
                id=data['id'],
                points=points,
                relay_id=data['relay_id']
            )
        else:
            raise ValueError("Zone data must contain either 'points' or 'rect'")


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
