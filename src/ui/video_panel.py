
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
        # Notify parent to update zone editor if needed
        self.update()  # ADD THIS LINE
    
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
