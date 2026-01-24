
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
        
        # Person detections (bounding boxes)
        self.persons = []
        
        # Zone violations for highlighting
        self.zone_violations = {}
        
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
        if frame is None or frame.size == 0:
            return
        
        self.current_frame = frame
        self._render_frame()
    
    def _render_frame(self) -> None:
        """Render frame with zones overlay and person bounding boxes."""
        if self.current_frame is None:
            return
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        # Draw person bounding boxes from detection results
        # Store these for reference: self.persons = [(x1, y1, x2, y2), ...]
        if hasattr(self, 'persons') and self.persons:
            for x1, y1, x2, y2 in self.persons:
                # Draw bounding box
                cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                # Draw person label
                cv2.putText(
                    frame_rgb,
                    "Person",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )
        
        # Draw zones on frame (now polygons instead of rectangles)
        for zone_id, points, color in self.zones:
            if len(points) < 2:
                continue
            
            # Check if this zone is currently violated - if so, use bright red with high visibility
            is_violated = hasattr(self, 'zone_violations') and zone_id in self.zone_violations
            if is_violated:
                draw_color = (255, 0, 0)  # Bright red for violated zones (RGB format since frame is RGB)
                border_width = 4  # Thicker border to grab attention
                fill_opacity = 0.35  # Higher opacity to make violation obvious
            else:
                draw_color = color  # Use default color if not violated
                border_width = 2
                fill_opacity = 0.15
            
            # Convert points to numpy array for cv2.polylines
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw polygon outline
            cv2.polylines(frame_rgb, [pts], True, draw_color, border_width)
            
            # Fill with semi-transparency
            overlay = frame_rgb.copy()
            cv2.fillPoly(overlay, [pts], draw_color)
            cv2.addWeighted(overlay, fill_opacity, frame_rgb, 1 - fill_opacity, 0, frame_rgb)
            
            # Draw zone label at first point
            if len(points) > 0:
                label = f"Zone {zone_id}"
                if is_violated:
                    label += " [VIOLATION]"
                cv2.putText(
                    frame_rgb,
                    label,
                    (int(points[0][0]) + 5, int(points[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 if is_violated else 0.5,
                    draw_color,
                    3 if is_violated else 2
                )
        
        # Convert to QPixmap with aspect ratio preservation
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget while preserving aspect ratio
        widget_size = self.video_label.size()
        pixmap = QPixmap.fromImage(q_image)
        
        # Safety check: if widget hasn't been sized yet, use a fallback
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            widget_size.setWidth(max(640, self.processing_width))
            widget_size.setHeight(max(360, self.processing_height))
        
        # Calculate scaling to fit widget
        scaled_pixmap = pixmap.scaled(
            widget_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Calculate offsets for letterbox/pillarbox
        self.offset_x = (widget_size.width() - scaled_pixmap.width()) // 2
        self.offset_y = (widget_size.height() - scaled_pixmap.height()) // 2
        self.scale = scaled_pixmap.width() / width if width > 0 else 1.0
        
        self.display_pixmap = scaled_pixmap
        self.video_label.setPixmap(scaled_pixmap)
        # Notify parent to update zone editor if needed
        self.update()  # ADD THIS LINE
    
    def set_zones(self, zones: list) -> None:
        """Set zones to display.
        
        Args:
            zones: List of (zone_id, points, color) tuples where points is List[Tuple[int, int]]
        """
        self.zones = zones
        if self.current_frame is not None:
            self._render_frame()
    
    def set_persons(self, persons: list) -> None:
        """Set detected persons to display.
        
        Args:
            persons: List of (x1, y1, x2, y2) bounding boxes
        """
        self.persons = persons
    
    def set_zone_violations(self, violations: dict) -> None:
        """Set which zones have violations to highlight them in red.
        
        Args:
            violations: Dict of {zone_id: True/False}
        """
        self.zone_violations = violations
    
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
