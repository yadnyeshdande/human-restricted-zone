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