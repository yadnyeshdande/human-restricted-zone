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
                logger.error(f"  URL: {self.rtsp_url}")
                logger.error(f"  Check: 1) URL format is correct")
                logger.error(f"  Check: 2) Username:password are correct (no double colons)")
                logger.error(f"  Check: 3) Camera IP and port are accessible")
                logger.error(f"  Check: 4) Camera may block connections after multiple failed attempts")
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
