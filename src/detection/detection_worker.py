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
