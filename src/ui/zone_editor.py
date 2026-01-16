
# =============================================================================
# File: ui/zone_editor.py
# =============================================================================
"""Zone drawing and editing widget."""

from typing import Optional, List, Tuple
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPaintEvent, QMouseEvent
from utils.logger import get_logger

logger = get_logger("ZoneEditor")


class ZoneEditor(QWidget):
    """Interactive zone drawing and editing overlay."""
    
    # Signals
    zone_created = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    zone_modified = pyqtSignal(int, tuple)  # zone_id, (x1, y1, x2, y2)
    zone_selected = pyqtSignal(int)  # zone_id
    
    # States
    STATE_IDLE = 0
    STATE_DRAWING = 1
    STATE_MOVING = 2
    STATE_RESIZING = 3
    
    def __init__(self, parent=None):
        """Initialize zone editor."""
        super().__init__(parent)
        
        # State
        self.state = self.STATE_IDLE
        self.zones: List[Tuple[int, QRect, QColor]] = []  # (id, rect, color)
        self.selected_zone_id: Optional[int] = None
        self.resize_handle: Optional[str] = None  # 'tl', 'tr', 'bl', 'br'
        
        # Drawing
        self.draw_start: Optional[QPoint] = None
        self.draw_current: Optional[QPoint] = None
        
        # Moving
        self.move_offset: Optional[QPoint] = None
        
        # Undo/Redo
        self.history: List[List[Tuple[int, QRect, QColor]]] = []
        self.history_index = -1
        
        # Configuration
        self.handle_size = 8
        self.default_color = QColor(0, 255, 0)
        self.selected_color = QColor(255, 255, 0)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        # Make background transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
    
    def set_zones(self, zones: List[Tuple[int, Tuple[int, int, int, int], QColor]]) -> None:
        """Set zones to display and edit.
        
        Args:
            zones: List of (zone_id, (x1, y1, x2, y2), color)
        """
        self.zones = [
            (zone_id, QRect(x1, y1, x2 - x1, y2 - y1), color)
            for zone_id, (x1, y1, x2, y2), color in zones
        ]
        self._save_history()
        self.update()
    
    def add_zone(self, zone_id: int, rect: Tuple[int, int, int, int], color: QColor = None) -> None:
        """Add a zone.
        
        Args:
            zone_id: Zone identifier
            rect: (x1, y1, x2, y2)
            color: Zone color
        """
        x1, y1, x2, y2 = rect
        self.zones.append((
            zone_id,
            QRect(x1, y1, x2 - x1, y2 - y1),
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
    
    def get_zones(self) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """Get all zones.
        
        Returns:
            List of (zone_id, (x1, y1, x2, y2))
        """
        return [
            (zone_id, (rect.x(), rect.y(), rect.right(), rect.bottom()))
            for zone_id, rect, _ in self.zones
        ]
    
    def _save_history(self) -> None:
        """Save current state to history."""
        # Remove any redo history
        self.history = self.history[:self.history_index + 1]
        
        # Save current state
        self.history.append([
            (zid, QRect(rect), QColor(color))
            for zid, rect, color in self.zones
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
                (zid, QRect(rect), QColor(color))
                for zid, rect, color in self.history[self.history_index]
            ]
            self.update()
            return True
        return False
    
    def redo(self) -> bool:
        """Redo last undone operation."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.zones = [
                (zid, QRect(rect), QColor(color))
                for zid, rect, color in self.history[self.history_index]
            ]
            self.update()
            return True
        return False
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint zones and handles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw existing zones
        for zone_id, rect, color in self.zones:
            is_selected = (zone_id == self.selected_zone_id)
            
            # Draw rectangle
            pen_width = 3 if is_selected else 2
            pen_color = self.selected_color if is_selected else color
            painter.setPen(QPen(pen_color, pen_width))
            painter.setBrush(QBrush(QColor(pen_color.red(), pen_color.green(), pen_color.blue(), 30)))
            painter.drawRect(rect)
            
            # Draw zone label
            painter.setPen(QPen(pen_color, 1))
            painter.drawText(rect.x() + 5, rect.y() + 15, f"Zone {zone_id}")
            
            # Draw resize handles for selected zone
            if is_selected:
                self._draw_handles(painter, rect)
        
        # Draw current drawing
        if self.state == self.STATE_DRAWING and self.draw_start and self.draw_current:
            painter.setPen(QPen(self.default_color, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            rect = QRect(self.draw_start, self.draw_current).normalized()
            painter.drawRect(rect)
    
    def _draw_handles(self, painter: QPainter, rect: QRect) -> None:
        """Draw resize handles."""
        handle_color = self.selected_color
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(Qt.white, 1))
        
        handles = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]
        
        for handle_pos in handles:
            handle_rect = QRect(
                handle_pos.x() - self.handle_size // 2,
                handle_pos.y() - self.handle_size // 2,
                self.handle_size,
                self.handle_size
            )
            painter.drawRect(handle_rect)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if event.button() != Qt.LeftButton:
            return
        
        pos = event.pos()
        
        # Check if clicking on resize handle
        if self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect:
                handle = self._get_handle_at(pos, selected_rect)
                if handle:
                    self.state = self.STATE_RESIZING
                    self.resize_handle = handle
                    return
        
        # Check if clicking on existing zone
        clicked_zone = self._get_zone_at(pos)
        if clicked_zone is not None:
            self.selected_zone_id = clicked_zone
            self.zone_selected.emit(clicked_zone)
            self.state = self.STATE_MOVING
            selected_rect = self._get_zone_rect(clicked_zone)
            if selected_rect:
                self.move_offset = pos - selected_rect.topLeft()
            self.update()
            return
        
        # Start drawing new zone
        self.selected_zone_id = None
        self.state = self.STATE_DRAWING
        self.draw_start = pos
        self.draw_current = pos
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        pos = event.pos()
        
        if self.state == self.STATE_DRAWING:
            self.draw_current = pos
            self.update()
        
        elif self.state == self.STATE_MOVING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect and self.move_offset:
                new_pos = pos - self.move_offset
                selected_rect.moveTo(new_pos)
                self.update()
        
        elif self.state == self.STATE_RESIZING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect and self.resize_handle:
                self._resize_rect(selected_rect, pos, self.resize_handle)
                self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() != Qt.LeftButton:
            return
        
        if self.state == self.STATE_DRAWING:
            if self.draw_start and self.draw_current:
                rect = QRect(self.draw_start, self.draw_current).normalized()
                if rect.width() > 10 and rect.height() > 10:
                    self.zone_created.emit((
                        rect.x(), rect.y(), rect.right(), rect.bottom()
                    ))
            self.draw_start = None
            self.draw_current = None
        
        elif self.state == self.STATE_MOVING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect:
                self.zone_modified.emit(
                    self.selected_zone_id,
                    (selected_rect.x(), selected_rect.y(), selected_rect.right(), selected_rect.bottom())
                )
                self._save_history()
        
        elif self.state == self.STATE_RESIZING and self.selected_zone_id is not None:
            selected_rect = self._get_zone_rect(self.selected_zone_id)
            if selected_rect:
                self.zone_modified.emit(
                    self.selected_zone_id,
                    (selected_rect.x(), selected_rect.y(), selected_rect.right(), selected_rect.bottom())
                )
                self._save_history()
        
        self.state = self.STATE_IDLE
        self.update()
    
    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() == Qt.Key_Delete and self.selected_zone_id is not None:
            self.remove_zone(self.selected_zone_id)
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self.redo()
    
    def _get_zone_at(self, pos: QPoint) -> Optional[int]:
        """Get zone ID at position."""
        for zone_id, rect, _ in reversed(self.zones):
            if rect.contains(pos):
                return zone_id
        return None
    
    def _get_zone_rect(self, zone_id: int) -> Optional[QRect]:
        """Get zone rectangle by ID."""
        for zid, rect, _ in self.zones:
            if zid == zone_id:
                return rect
        return None
    
    def _get_handle_at(self, pos: QPoint, rect: QRect) -> Optional[str]:
        """Get resize handle at position."""
        handles = {
            'tl': rect.topLeft(),
            'tr': rect.topRight(),
            'bl': rect.bottomLeft(),
            'br': rect.bottomRight()
        }
        
        for handle_name, handle_pos in handles.items():
            handle_rect = QRect(
                handle_pos.x() - self.handle_size,
                handle_pos.y() - self.handle_size,
                self.handle_size * 2,
                self.handle_size * 2
            )
            if handle_rect.contains(pos):
                return handle_name
        return None
    
    def _resize_rect(self, rect: QRect, pos: QPoint, handle: str) -> None:
        """Resize rectangle by dragging handle."""
        if handle == 'tl':
            rect.setTopLeft(pos)
        elif handle == 'tr':
            rect.setTopRight(pos)
        elif handle == 'bl':
            rect.setBottomLeft(pos)
        elif handle == 'br':
            rect.setBottomRight(pos)
        
        # Ensure minimum size
        if rect.width() < 10:
            rect.setWidth(10)
        if rect.height() < 10:
            rect.setHeight(10)
