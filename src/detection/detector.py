# =============================================================================
# File: detection/detector.py
# =============================================================================
"""YOLO detector wrapper."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("Detector")


class PersonDetector:
    """YOLO-based person detector."""
    
    PERSON_CLASS_ID = 0
    MODELS_DIR = Path(__file__).parent.parent.parent / "models"  # Project root/models/
    
    def __init__(self, model_name: str = None, conf_threshold: float = None):
        """Initialize detector.
        
        Args:
            model_name: YOLO model name (None = use app_settings)
            conf_threshold: Confidence threshold (None = use app_settings)
        """
        from config.app_settings import SETTINGS
        
        self.conf_threshold = conf_threshold or SETTINGS.detection_confidence
        model_name = model_name or SETTINGS.yolo_model
        self.model = None
        self.device = 'cpu'
        self.model_path = None
        self.model_loaded = False
        self.model_name = model_name
        
        logger.info(f"Initializing YOLO detector: {model_name}, confidence: {self.conf_threshold}")
        
        # Ensure models directory exists
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            from ultralytics import YOLO
            
            # Check if model exists in models folder
            model_file = self.MODELS_DIR / model_name
            
            if model_file.exists():
                logger.info(f"[OK] Model found in local storage: {model_file}")
                size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"    Model size: {size_mb:.1f} MB")
                self.model_path = str(model_file)
                
                logger.info(f"Loading model: {model_name}...")
                self.model = YOLO(self.model_path)
                self.model_loaded = True
                logger.info(f"[OK] Model loaded successfully: {model_name}")
            else:
                logger.warning(f"Model not found in: {self.MODELS_DIR}")
                logger.warning(f"Expected path: {model_file}")
                logger.info(f"Attempting to download model: {model_name}")
                
                # Try to load from ultralytics (will download if needed)
                try:
                    logger.info(f"Downloading {model_name} from ultralytics...")
                    self.model = YOLO(model_name)
                    self.model_loaded = True
                    logger.info(f"[OK] Model loaded from ultralytics: {model_name}")
                    logger.warning(f"NOTE: Model was auto-downloaded. For production use,")
                    logger.warning(f"      download the model and place it in: {self.MODELS_DIR}")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    raise
            
            # Try to use CUDA if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.info("[OK] YOLO using GPU (CUDA)")
                else:
                    logger.info("[OK] YOLO using CPU")
            except ImportError:
                logger.info("[OK] YOLO using CPU (PyTorch not available)")
                
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            self.model_loaded = False
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded successfully."""
        return self.model_loaded and self.model is not None
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons in frame.
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.model is None or not self.model_loaded:
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
