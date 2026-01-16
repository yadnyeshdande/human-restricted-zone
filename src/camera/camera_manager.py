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