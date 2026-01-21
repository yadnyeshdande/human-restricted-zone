# ============================================================================
"""Polygon zone drawing and editing widget."""
# =============================================================================
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPaintEvent, QMouseEvent, QPolygon
from utils.logger import get_logger

logger = get_logger("ZoneEditor")


class ZoneEditor(QWidget):
    """Interactive polygon zone drawing and editing overlay."""
    
    # Signals
    zone_created = pyqtSignal(list)  # List of (x, y) points
    zone_modified = pyqtSignal(int, list)  # zone_id, List of (x, y) points
    zone_selected = pyqtSignal(int)  # zone_id
    
    # States
    STATE_IDLE = 0
    STATE_DRAWING = 1
    STATE_MOVING_ZONE = 2
    STATE_MOVING_POINT = 3
    
    def __init__(self, parent=None):
        """Initialize zone editor."""
        super().__init__(parent)
        
        # State
        self.state = self.STATE_IDLE
        self.zones: List[Tuple[int, List[Tuple[int, int]], QColor]] = []  # (id, points, color)
        self.selected_zone_id: Optional[int] = None
        self.selected_point_index: Optional[int] = None
        
        # Drawing
        self.current_points: List[Tuple[int, int]] = []  # Points being drawn
        
        # Moving
        self.move_offset: Optional[QPoint] = None
        
        # Undo/Redo
        self.history: List[List[Tuple[int, List[Tuple[int, int]], QColor]]] = []
        self.history_index = -1
        
        # Configuration
        self.point_radius = 6  # Radius for point handles
        self.click_threshold = 15  # Distance to close polygon
        self.default_color = QColor(0, 255, 0)
        self.selected_color = QColor(255, 255, 0)
        self.drawing_color = QColor(255, 128, 0)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        # Make background transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
    
    def set_zones(self, zones: List[Tuple[int, List[Tuple[int, int]], QColor]]) -> None:
        """Set zones to display and edit.
        
        Args:
            zones: List of (zone_id, points, color)
        """
        self.zones = [(zone_id, list(points), color) for zone_id, points, color in zones]
        self._save_history()
        self.update()
    
    def add_zone(self, zone_id: int, points: List[Tuple[int, int]], color: QColor = None) -> None:
        """Add a zone.
        
        Args:
            zone_id: Zone identifier
            points: List of (x, y) points
            color: Zone color
        """
        self.zones.append((
            zone_id,
            list(points),
            color or self.default_color
        ))
        self._save_history()
        self.update()
    
    def remove_zone(self, zone_id: int) -> None:
        """Remove a zone."""
        self.zones = [z for z in self.zones if z[0] != zone_id]
        if self.selected_zone_id == zone_id:
            self.selected_zone_id = None
        self._save_history()
        self.update()
    
    def clear_zones(self) -> None:
        """Clear all zones."""
        self.zones.clear()
        self.selected_zone_id = None
        self._save_history()
        self.update()
    
    def get_zones(self) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """Get all zones.
        
        Returns:
            List of (zone_id, points)
        """
        return [(zone_id, list(points)) for zone_id, points, _ in self.zones]
    
    def _save_history(self) -> None:
        """Save current state to history."""
        # Remove any redo history
        self.history = self.history[:self.history_index + 1]
        
        # Save current state (deep copy)
        self.history.append([
            (zid, list(points), QColor(color))
            for zid, points, color in self.zones
        ])
        self.history_index = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self) -> bool:
        """Undo last operation."""
        if self.history_index > 0:
            self.history_index -= 1
            self.zones = [
                (zid, list(points), QColor(color))
                for zid, points, color in self.history[self.history_index]
            ]
            self.update()
            return True
        return False
    
    def redo(self) -> bool:
        """Redo last undone operation."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.zones = [
                (zid, list(points), QColor(color))
                for zid, points, color in self.history[self.history_index]
            ]
            self.update()
            return True
        return False
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint zones and handles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw existing zones
        for zone_id, points, color in self.zones:
            if len(points) < 2:
                continue
            
            is_selected = (zone_id == self.selected_zone_id)
            
            # Draw polygon
            pen_width = 3 if is_selected else 2
            pen_color = self.selected_color if is_selected else color
            painter.setPen(QPen(pen_color, pen_width))
            painter.setBrush(QBrush(QColor(pen_color.red(), pen_color.green(), pen_color.blue(), 30)))
            
            # Create QPolygon for drawing
            qpoly = QPolygon([QPoint(int(x), int(y)) for x, y in points])
            painter.drawPolygon(qpoly)
            
            # Draw zone label at first point
            if points:
                painter.setPen(QPen(pen_color, 1))
                painter.drawText(int(points[0][0]) + 5, int(points[0][1]) - 5, f"Zone {zone_id}")
            
            # Draw point handles for selected zone
            if is_selected:
                self._draw_point_handles(painter, points, pen_color)
        
        # Draw current drawing polygon
        if self.state == self.STATE_DRAWING and len(self.current_points) > 0:
            painter.setPen(QPen(self.drawing_color, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            
            # Draw lines between points
            for i in range(len(self.current_points) - 1):
                p1 = QPoint(int(self.current_points[i][0]), int(self.current_points[i][1]))
                p2 = QPoint(int(self.current_points[i + 1][0]), int(self.current_points[i + 1][1]))
                painter.drawLine(p1, p2)
            
            # Draw points
            for px, py in self.current_points:
                painter.setBrush(QBrush(self.drawing_color))
                painter.drawEllipse(QPoint(int(px), int(py)), self.point_radius, self.point_radius)
            
            # Draw instruction
            if len(self.current_points) >= 3:
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(10, 30, "Click near first point to close polygon, or Right-click to finish")
    
    def _draw_point_handles(self, painter: QPainter, points: List[Tuple[int, int]], color: QColor) -> None:
        """Draw handles for polygon points."""
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.white, 2))
        
        for px, py in points:
            painter.drawEllipse(QPoint(int(px), int(py)), self.point_radius, self.point_radius)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        pos = event.pos()
        
        if event.button() == Qt.LeftButton:
            if self.state == self.STATE_IDLE:
                # Check if clicking on point of selected zone
                if self.selected_zone_id is not None:
                    point_index = self._get_point_at(pos, self.selected_zone_id)
                    if point_index is not None:
                        self.state = self.STATE_MOVING_POINT
                        self.selected_point_index = point_index
                        return
                
                # Check if clicking inside existing zone
                clicked_zone = self._get_zone_at(pos)
                if clicked_zone is not None:
                    self.selected_zone_id = clicked_zone
                    self.zone_selected.emit(clicked_zone)
                    self.state = self.STATE_MOVING_ZONE
                    
                    # Calculate offset from first point
                    zone_points = self._get_zone_points(clicked_zone)
                    if zone_points and len(zone_points) > 0:
                        first_point = zone_points[0]
                        self.move_offset = QPoint(
                            pos.x() - int(first_point[0]),
                            pos.y() - int(first_point[1])
                        )
                    self.update()
                    return
                
                # Start drawing new zone
                self.selected_zone_id = None
                self.state = self.STATE_DRAWING
                self.current_points = [(pos.x(), pos.y())]
                self.update()
            
            elif self.state == self.STATE_DRAWING:
                # Check if closing polygon (click near first point)
                if len(self.current_points) >= 3:
                    first_point = self.current_points[0]
                    dist = ((pos.x() - first_point[0])**2 + (pos.y() - first_point[1])**2)**0.5
                    
                    if dist < self.click_threshold:
                        # Close polygon
                        self._finish_polygon()
                        return
                
                # Add point to current polygon
                self.current_points.append((pos.x(), pos.y()))
                self.update()
        
        elif event.button() == Qt.RightButton:
            if self.state == self.STATE_DRAWING:
                # Finish polygon without closing
                if len(self.current_points) >= 3:
                    self._finish_polygon()
                else:
                    # Cancel drawing
                    self.current_points = []
                    self.state = self.STATE_IDLE
                    self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        pos = event.pos()
        
        if self.state == self.STATE_MOVING_ZONE and self.selected_zone_id is not None:
            zone_points = self._get_zone_points(self.selected_zone_id)
            if zone_points and self.move_offset:
                # Calculate new position
                new_x = pos.x() - self.move_offset.x()
                new_y = pos.y() - self.move_offset.y()
                
                # Calculate offset from original position
                if len(zone_points) > 0:
                    offset_x = new_x - zone_points[0][0]
                    offset_y = new_y - zone_points[0][1]
                    
                    # Move all points
                    for i in range(len(zone_points)):
                        zone_points[i] = (
                            zone_points[i][0] + offset_x,
                            zone_points[i][1] + offset_y
                        )
                
                self.update()
        
        elif self.state == self.STATE_MOVING_POINT and self.selected_zone_id is not None:
            zone_points = self._get_zone_points(self.selected_zone_id)
            if zone_points and self.selected_point_index is not None:
                if 0 <= self.selected_point_index < len(zone_points):
                    zone_points[self.selected_point_index] = (pos.x(), pos.y())
                    self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() != Qt.LeftButton:
            return
        
        if self.state == self.STATE_MOVING_ZONE and self.selected_zone_id is not None:
            zone_points = self._get_zone_points(self.selected_zone_id)
            if zone_points:
                self.zone_modified.emit(self.selected_zone_id, zone_points)
                self._save_history()
        
        elif self.state == self.STATE_MOVING_POINT and self.selected_zone_id is not None:
            zone_points = self._get_zone_points(self.selected_zone_id)
            if zone_points:
                self.zone_modified.emit(self.selected_zone_id, zone_points)
                self._save_history()
        
        if self.state != self.STATE_DRAWING:
            self.state = self.STATE_IDLE
        
        self.update()
    
    def _finish_polygon(self) -> None:
        """Finish drawing current polygon."""
        if len(self.current_points) >= 3:
            self.zone_created.emit(self.current_points)
        
        self.current_points = []
        self.state = self.STATE_IDLE
        self.update()
    
    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() == Qt.Key_Delete and self.selected_zone_id is not None:
            self.remove_zone(self.selected_zone_id)
        elif event.key() == Qt.Key_Escape:
            # Cancel current drawing
            if self.state == self.STATE_DRAWING:
                self.current_points = []
                self.state = self.STATE_IDLE
                self.update()
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self.redo()
    
    def _get_zone_at(self, pos: QPoint) -> Optional[int]:
        """Get zone ID at position using point-in-polygon test."""
        for zone_id, points, _ in reversed(self.zones):
            if len(points) < 3:
                continue
            
            if self._point_in_polygon(pos, points):
                return zone_id
        return None
    
    def _point_in_polygon(self, point: QPoint, polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon."""
        x, y = point.x(), point.y()
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _get_zone_points(self, zone_id: int) -> Optional[List[Tuple[int, int]]]:
        """Get zone points by ID (returns reference, not copy)."""
        for zid, points, _ in self.zones:
            if zid == zone_id:
                return points
        return None
    
    def _get_point_at(self, pos: QPoint, zone_id: int) -> Optional[int]:
        """Get index of point near position."""
        zone_points = self._get_zone_points(zone_id)
        if not zone_points:
            return None
        
        threshold = self.point_radius + 5
        
        for i, (px, py) in enumerate(zone_points):
            dist = ((pos.x() - px)**2 + (pos.y() - py)**2)**0.5
            if dist < threshold:
                return i
        
        return None