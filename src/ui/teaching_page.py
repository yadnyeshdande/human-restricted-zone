# =============================================================================
# File: ui/teaching_page.py
# =============================================================================
"""Teaching mode interface for zone editing."""

import queue
from typing import Dict, List, Tuple, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QInputDialog, QMessageBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from .video_panel import VideoPanel
from .zone_editor import ZoneEditor
from utils.logger import get_logger

from ui import video_panel

logger = get_logger("TeachingPage")


class TeachingPage(QWidget):
    """Zone editor interface with multi-camera support."""
    
    zones_changed = pyqtSignal()
    
    def __init__(
        self,
        config_manager: ConfigManager,
        camera_manager: CameraManager,
        parent=None
    ):
        """Initialize teaching page.
        
        Args:
            config_manager: Configuration manager
            camera_manager: Camera manager
            parent: Parent widget
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.camera_manager = camera_manager
        
        # Video panels and editors
        self.video_panels: Dict[int, VideoPanel] = {}
        self.zone_editors: Dict[int, ZoneEditor] = {}
        self.panel_containers: Dict[int, QWidget] = {}
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_frames)
        self.update_timer.start(33)  # ~30 FPS
        
        self._setup_ui()
        self._load_cameras()
    
    def cleanup(self) -> None:
        """Cleanup resources before shutdown.
        
        Stops the update timer. CRITICAL for preventing CPU spin.
        """
        logger.info("TeachingPage: Starting cleanup...")
        
        # Stop the update timer (CRITICAL - prevents CPU waste)
        if hasattr(self, 'update_timer'):
            try:
                self.update_timer.stop()
                logger.debug("  OK: Update timer stopped")
            except Exception as e:
                logger.warning(f"Error stopping timer: {e}")
        
        logger.info("  OK: TeachingPage cleanup complete")
    
    def __del__(self):
        """Destructor for automatic cleanup."""
        try:
            if hasattr(self, 'update_timer'):
                self.update_timer.stop()
        except:
            pass
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        
        self.add_camera_btn = QPushButton("Add Camera")
        self.add_camera_btn.clicked.connect(self._add_camera)
        toolbar.addWidget(self.add_camera_btn)
        
        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.clicked.connect(self._save_configuration)
        toolbar.addWidget(self.save_btn)
        
        self.clear_zones_btn = QPushButton("Clear All Zones")
        self.clear_zones_btn.clicked.connect(self._clear_all_zones)
        toolbar.addWidget(self.clear_zones_btn)
        
        toolbar.addStretch()
        
        self.status_label = QLabel("Ready")
        toolbar.addWidget(self.status_label)
        
        layout.addLayout(toolbar)
        
        # Camera grid (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.camera_grid_widget = QWidget()
        self.camera_grid = QGridLayout(self.camera_grid_widget)
        self.camera_grid.setSpacing(10)
        # Stretches will be set dynamically in _add_camera_panel
        scroll_area.setWidget(self.camera_grid_widget)
        
        layout.addWidget(scroll_area)
        
        # Instructions
        instructions = QLabel(
            "Click to add points and create polygon zones. "
            "Click near first point or right-click to finish polygon. "
            "Click zone to select, drag to move entire zone. "
            "Drag individual points to reshape. "
            "Press Delete to remove selected zone, Esc to cancel drawing. "
            "Ctrl+Z to undo, Ctrl+Y to redo."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(instructions)
    
    def _load_cameras(self) -> None:
        """Load cameras from configuration."""
        cameras = self.config_manager.get_all_cameras()
        for camera in cameras:
            self._add_camera_panel(camera.id, camera.rtsp_url)
            
            # Load zones (now using points instead of rect)
            for zone in camera.zones:
                self._add_zone_visual(camera.id, zone.id, zone.points, zone.relay_id)
    
    def _add_camera(self) -> None:
        """Add a new camera."""
        rtsp_url, ok = QInputDialog.getText(
            self,
            "Add Camera",
            "Enter RTSP URL:\n(e.g., rtsp://admin:Pass_123@192.168.1.64:554/stream)",
            text="rtsp://"
        )
        
        if not ok or not rtsp_url:
            return
        
        # Add to configuration
        camera = self.config_manager.add_camera(rtsp_url)
        
        # Start camera capture
        if self.camera_manager.add_camera(camera.id, rtsp_url):
            self._add_camera_panel(camera.id, rtsp_url)
            self.status_label.setText(f"Camera {camera.id} added")
            logger.info(f"Camera {camera.id} added: {rtsp_url}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to connect to camera")
    
    def _add_camera_panel(self, camera_id: int, rtsp_url: str) -> None:
        """Add camera panel to grid.
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP URL
        """
        # Calculate grid position
        num_cameras = len(self.video_panels)
        cols = 2  # 2 columns
        row = num_cameras // cols
        col = num_cameras % cols
        
        # Dynamically set stretch factors for new row/column
        self.camera_grid.setRowStretch(row, 1)
        self.camera_grid.setColumnStretch(col, 1)
        
        # Create container
        
        # Create container
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create video panel with proper aspect ratio
        video_panel = VideoPanel(
            camera_id=camera_id,
            processing_resolution=self.config_manager.config.processing_resolution
        )
        video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Calculate minimum size based on 16:9 aspect ratio
        # Minimum width 480px means minimum height should be 480/16*9 = 270px
        # But add extra for controls, so use 350px minimum height
        video_panel.setMinimumSize(480, 350)
        
        # Create zone editor overlay - MUST be child of video_label, not video_panel
        # Create zone editor overlay - MUST be child of video_label
        zone_editor = ZoneEditor(video_panel.video_label)
        zone_editor.setGeometry(0, 0, 
                                video_panel.video_label.width(),
                                video_panel.video_label.height())
        zone_editor.raise_()  # Ensure overlay is on top
        zone_editor.zone_created.connect(
            lambda rect, cid=camera_id: self._on_zone_created(cid, rect)
        )
        zone_editor.zone_modified.connect(
            lambda zid, rect, cid=camera_id: self._on_zone_modified(cid, zid, rect)
        )
        
        container_layout.addWidget(video_panel)
        
        # Camera controls
        controls = QHBoxLayout()
        
        remove_btn = QPushButton(f"Remove Camera {camera_id}")
        remove_btn.clicked.connect(lambda: self._remove_camera(camera_id))
        controls.addWidget(remove_btn)
        
        delete_zone_btn = QPushButton("Delete Selected Zone")
        delete_zone_btn.clicked.connect(lambda: self._delete_selected_zone(camera_id))
        controls.addWidget(delete_zone_btn)
        
        controls.addStretch()
        container_layout.addLayout(controls)
        
        # Add to grid
        self.camera_grid.addWidget(container, row, col)
        
        # Store references
        self.video_panels[camera_id] = video_panel
        self.zone_editors[camera_id] = zone_editor
        self.panel_containers[camera_id] = container
        
        logger.info(f"Camera panel {camera_id} added to UI")
    
    def _remove_camera(self, camera_id: int) -> None:
        """Remove camera."""
        reply = QMessageBox.question(
            self,
            "Remove Camera",
            f"Remove camera {camera_id} and all its zones?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove from managers
            self.camera_manager.remove_camera(camera_id)
            self.config_manager.remove_camera(camera_id)
            
            # Remove from UI
            if camera_id in self.panel_containers:
                container = self.panel_containers[camera_id]
                self.camera_grid.removeWidget(container)
                container.deleteLater()
                
                del self.video_panels[camera_id]
                del self.zone_editors[camera_id]
                del self.panel_containers[camera_id]
            
            self._reorganize_grid()
            self.status_label.setText(f"Camera {camera_id} removed")
            self.zones_changed.emit()
    
    def _reorganize_grid(self) -> None:
        """Reorganize camera grid after removal."""
        # Remove all widgets
        for camera_id, container in self.panel_containers.items():
            self.camera_grid.removeWidget(container)
        
        # Re-add in order
        cols = 2
        for idx, (camera_id, container) in enumerate(sorted(self.panel_containers.items())):
            row = idx // cols
            col = idx % cols
            self.camera_grid.addWidget(container, row, col)
    
    def _on_zone_created(self, camera_id: int, points: List[Tuple[int, int]]) -> None:
        """Handle zone creation."""
        # Convert from widget coordinates to processing coordinates
        if camera_id in self.video_panels:
            video_panel = self.video_panels[camera_id]
            
            # Convert all points
            processing_points = [
                video_panel.widget_to_processing(x, y)
                for x, y in points
            ]
            
            logger.info(f"Polygon zone created - {len(points)} points")
        else:
            processing_points = points
        
        # Add to configuration with processing coordinates
        zone = self.config_manager.add_zone(camera_id, processing_points)
        
        if zone:
            self._add_zone_visual(camera_id, zone.id, processing_points, zone.relay_id)
            self.status_label.setText(
                f"Zone {zone.id} created for Camera {camera_id}, "
                f"assigned to Relay {zone.relay_id}"
            )
            self.zones_changed.emit()
            logger.info(
                f"Zone created: Camera {camera_id}, Zone {zone.id}, "
                f"Relay {zone.relay_id}, Points: {len(processing_points)}"
            )
    
    def _on_zone_modified(self, camera_id: int, zone_id: int, points: List[Tuple[int, int]]) -> None:
        """Handle zone modification."""
        # Convert from widget coordinates to processing coordinates
        if camera_id in self.video_panels:
            video_panel = self.video_panels[camera_id]
            
            processing_points = [
                video_panel.widget_to_processing(x, y)
                for x, y in points
            ]
        else:
            processing_points = points
        
        if self.config_manager.update_zone(camera_id, zone_id, processing_points):
            self._update_zone_visuals(camera_id)
            self.status_label.setText(f"Zone {zone_id} updated")
            self.zones_changed.emit()
    
    def _add_zone_visual(
        self,
        camera_id: int,
        zone_id: int,
        points: List[Tuple[int, int]],
        relay_id: int
    ) -> None:
        """Add zone to visual editor."""
        if camera_id in self.zone_editors and camera_id in self.video_panels:
            # Convert processing coordinates to widget coordinates for display
            video_panel = self.video_panels[camera_id]
            
            widget_points = [
                video_panel.processing_to_widget(x, y)
                for x, y in points
            ]
            
            color = self._get_zone_color(relay_id)
            self.zone_editors[camera_id].add_zone(zone_id, widget_points, color)
            self._update_zone_visuals(camera_id)
    
    def _update_zone_visuals(self, camera_id: int) -> None:
        """Update zone visuals on video panel."""
        if camera_id not in self.video_panels or camera_id not in self.zone_editors:
            return
        
        zones = self.zone_editors[camera_id].get_zones()
        
        # Get relay IDs from config
        camera = self.config_manager.get_camera(camera_id)
        if not camera:
            return
        
        zone_data = []
        for zone_id, points in zones:  # Now returns points instead of rect
            # Find relay_id
            relay_id = None
            for zone in camera.zones:
                if zone.id == zone_id:
                    relay_id = zone.relay_id
                    break
            
            if relay_id:
                color = self._get_zone_color(relay_id)
                zone_data.append((zone_id, points, (color.red(), color.green(), color.blue())))
        
        self.video_panels[camera_id].set_zones(zone_data)
    
    def _get_zone_color(self, relay_id: int) -> QColor:
        """Get color for relay ID."""
        colors = [
            QColor(0, 255, 0),      # Green
            QColor(0, 255, 255),    # Cyan
            QColor(255, 0, 255),    # Magenta
            QColor(255, 255, 0),    # Yellow
            QColor(255, 128, 0),    # Orange
            QColor(128, 0, 255),    # Purple
        ]
        return colors[(relay_id - 1) % len(colors)]
    
    def _delete_selected_zone(self, camera_id: int) -> None:
        """Delete selected zone for camera."""
        if camera_id not in self.zone_editors:
            return
        
        zone_id = self.zone_editors[camera_id].selected_zone_id
        if zone_id is not None:
            self.config_manager.remove_zone(camera_id, zone_id)
            self.zone_editors[camera_id].remove_zone(zone_id)
            self._update_zone_visuals(camera_id)
            self.status_label.setText(f"Zone {zone_id} deleted")
            self.zones_changed.emit()
    
    def _clear_all_zones(self) -> None:
        """Clear all zones from all cameras."""
        reply = QMessageBox.question(
            self,
            "Clear All Zones",
            "Remove all zones from all cameras?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for camera in self.config_manager.get_all_cameras():
                camera.zones.clear()
            
            for zone_editor in self.zone_editors.values():
                zone_editor.clear_zones()
            
            for camera_id in self.video_panels.keys():
                self._update_zone_visuals(camera_id)
            
            self.status_label.setText("All zones cleared")
            self.zones_changed.emit()
    
    def _save_configuration(self) -> None:
        """Save configuration to file."""
        if self.config_manager.save():
            self.status_label.setText("Configuration saved")
            QMessageBox.information(self, "Success", "Configuration saved successfully")
        else:
            QMessageBox.warning(self, "Error", "Failed to save configuration")
    
    def _update_frames(self) -> None:
        """Update video frames and sync zone editor geometry."""
        for camera_id, video_panel in self.video_panels.items():
            frame = self.camera_manager.get_latest_frame(camera_id)
            if frame is not None:
                # 1. Update frame first (calculates new scale/offsets)
                video_panel.update_frame(frame)
                
                # 2. Sync ZoneEditor geometry to match video_label
                if camera_id in self.zone_editors:
                    zone_editor = self.zone_editors[camera_id]
                    target_rect = video_panel.video_label.geometry()
                    
                    # If video size changed, sync overlay and refresh zones
                    if zone_editor.geometry() != target_rect:
                        zone_editor.setGeometry(target_rect)
                        # Refresh zones with NEW scale from step 1
                        self._update_zone_visuals(camera_id)
                
                # Update info
                fps = self.camera_manager.get_fps(camera_id)
                connected = self.camera_manager.is_connected(camera_id)
                status = "Connected" if connected else "Disconnected"
                video_panel.update_info(f"Camera {camera_id} | {status} | {fps:.1f} FPS")
    
    def resizeEvent(self, event):
        """Handle resize event - timer will sync geometry."""
        super().resizeEvent(event)
        # The _update_frames timer will handle geometry sync automatically
        # This prevents race conditions with scale calculation
    
    # def resizeEvent(self, event):
    #     """Handle resize to update zone editor geometry and zone positions."""
    #     super().resizeEvent(event)
        
    #     for camera_id, video_panel in self.video_panels.items():
    #         if camera_id in self.zone_editors:
    #             zone_editor = self.zone_editors[camera_id]
                
    #             # Update zone editor size to match video label
    #             zone_editor.setGeometry(0, 0, 
    #                                 video_panel.video_label.width(), 
    #                                 video_panel.video_label.height())
                
    #             # Refresh zone visuals with updated coordinates
    #             self._update_zone_visuals(camera_id)
                    