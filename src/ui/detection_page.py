# =============================================================================
# File: ui/detection_page.py
# =============================================================================
"""Detection mode interface for live monitoring."""

import cv2
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from relay.relay_manager import RelayManager
from detection.detection_worker import DetectionWorker
from .video_panel import VideoPanel
from utils.logger import get_logger

logger = get_logger("DetectionPage")


class DetectionPage(QWidget):
    """Live detection interface."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        camera_manager: CameraManager,
        relay_manager: RelayManager,
        parent=None
    ):
        """Initialize detection page.
        
        Args:
            config_manager: Configuration manager
            camera_manager: Camera manager
            relay_manager: Relay manager
            parent: Parent widget
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.camera_manager = camera_manager
        self.relay_manager = relay_manager
        
        # Detection state
        self.is_running = False
        self.detection_workers: Dict[int, DetectionWorker] = {}
        self.detection_queues: Dict[int, queue.Queue] = {}
        
        # Video panels
        self.video_panels: Dict[int, VideoPanel] = {}
        self.panel_containers: Dict[int, QWidget] = {}
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(33)  # ~30 FPS
        
        # Snapshot directory
        self.snapshot_dir = Path("snapshots")
        self.snapshot_dir.mkdir(exist_ok=True)
        
        self._setup_ui()
        self._load_cameras()
    
    def cleanup(self) -> None:
        """Cleanup resources before shutdown.
        
        Stops the update timer. CRITICAL for preventing CPU spin.
        """
        logger.info("DetectionPage: Starting cleanup...")
        
        # Stop the update timer (CRITICAL - prevents CPU waste)
        if hasattr(self, 'update_timer'):
            try:
                self.update_timer.stop()
                logger.debug("  OK: Update timer stopped")
            except Exception as e:
                logger.warning(f"Error stopping timer: {e}")
        
        logger.info("  OK: DetectionPage cleanup complete")
    
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
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._start_detection)
        toolbar.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self._stop_detection)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)
        
        toolbar.addStretch()
        
        self.status_label = QLabel("Detection Stopped")
        self.status_label.setStyleSheet("font-weight: bold;")
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
        
        # Status panel
        self.stats_label = QLabel("Waiting to start...")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.stats_label)
    
    def _load_cameras(self) -> None:
        """Load cameras and zones from configuration."""
        cameras = self.config_manager.get_all_cameras()
        
        for camera in cameras:
            self._add_camera_panel(camera.id, camera.rtsp_url)
    
    def _add_camera_panel(self, camera_id: int, rtsp_url: str) -> None:
        """Add camera panel to grid."""
        # Calculate grid position
        num_cameras = len(self.video_panels)
        cols = 2  # 2 columns
        row = num_cameras // cols
        col = num_cameras % cols
        
        # Dynamically set stretch factors for new row/column
        self.camera_grid.setRowStretch(row, 1)
        self.camera_grid.setColumnStretch(col, 1)
        
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
        video_panel.setMinimumSize(480, 350)  # Changed from 400, 300
        
        container_layout.addWidget(video_panel)
        
        # Add to grid
        self.camera_grid.addWidget(container, row, col)
        
        # Store references
        self.video_panels[camera_id] = video_panel
        self.panel_containers[camera_id] = container
        
        # Load zones for display
        self._load_zones_for_camera(camera_id)
        
        logger.info(f"Detection panel {camera_id} added to UI")
    
    def _load_zones_for_camera(self, camera_id: int) -> None:
        """Load zones for a camera."""
        camera = self.config_manager.get_camera(camera_id)
        if not camera or camera_id not in self.video_panels:
            return
        
        zone_data = []
        for zone in camera.zones:
            color = self._get_zone_color(zone.relay_id)
            zone_data.append((
                zone.id,
                zone.points,  # Now using points instead of rect
                (color.red(), color.green(), color.blue())
            ))
        
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
    
    def _start_detection(self) -> None:
        """Start detection on all cameras."""
        if self.is_running:
            return
        
        logger.info("Starting detection...")
        
        # Show progress dialog
        from PyQt5.QtWidgets import QProgressDialog, QMessageBox
        from PyQt5.QtCore import Qt as QtCore
        
        progress = QProgressDialog(
            "Initializing detection system...\n\nLoading YOLO model...",
            None,
            0,
            0,
            self
        )
        progress.setWindowTitle("Starting Detection")
        progress.setWindowModality(QtCore.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumWidth(400)
        progress.setMinimumHeight(150)
        progress.show()
        
        # Force immediate repaint to ensure dialog is visible
        progress.repaint()
        
        try:
            cameras = self.config_manager.get_all_cameras()
            
            if not cameras:
                progress.close()
                QMessageBox.warning(self, "No Cameras", "No cameras configured. Please add cameras first.")
                return
            
            for idx, camera in enumerate(cameras):
                progress.setLabelText(
                    f"Initializing detection system...\n\n"
                    f"Loading YOLO model for camera {camera.id}..."
                )
                progress.repaint()
                
                # Get frame queue from camera manager
                frame_queue = self.camera_manager.get_frame_queue(camera.id)
                if not frame_queue:
                    logger.warning(f"No frame queue for camera {camera.id}")
                    continue
                
                # Prepare zones data (now with points instead of rect)
                zones_data = [
                    (zone.id, zone.points, zone.relay_id)
                    for zone in camera.zones
                ]
                
                # Create detection worker with proper error handling
                try:
                    worker = DetectionWorker(
                        camera_id=camera.id,
                        frame_queue=frame_queue,
                        zones=zones_data,
                        on_violation=self._handle_violation
                    )
                    
                    self.detection_workers[camera.id] = worker
                    worker.start()
                    logger.info(f"[OK] Detection started for camera {camera.id}")
                    
                except Exception as e:
                    progress.close()
                    logger.error(f"Failed to start detection for camera {camera.id}: {e}")
                    
                    # Check if it's a model loading error
                    from pathlib import Path
                    models_dir = Path("models")
                    model_file = models_dir / SETTINGS.yolo_model
                    
                    if not model_file.exists():
                        QMessageBox.critical(
                            self,
                            "Model Not Found",
                            f"YOLO model '{SETTINGS.yolo_model}' not found!\n\n"
                            f"Expected location: {model_file}\n\n"
                            f"To fix this:\n"
                            f"1. Go to Settings → Detection Settings\n"
                            f"2. Click 'Check & Download' button\n"
                            f"3. Wait for download to complete\n"
                            f"4. Restart the application\n\n"
                            f"Error details: {str(e)}"
                        )
                    else:
                        QMessageBox.critical(
                            self,
                            "Detection Startup Error",
                            f"Failed to initialize detection for camera {camera.id}:\n\n"
                            f"{str(e)}\n\n"
                            f"Check logs for more details."
                        )
                    return
            
            progress.close()
            
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("[OK] Detection Running")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
            logger.info("[OK] Detection system fully initialized")
            
        except Exception as e:
            progress.close()
            logger.error(f"Failed to start detection: {e}")
            QMessageBox.critical(
                self,
                "Detection Error",
                f"Failed to start detection:\n{e}"
            )
    
    def _stop_detection(self) -> None:
        """Stop detection on all cameras."""
        if not self.is_running:
            return
        
        logger.info("Stopping detection...")
        
        for camera_id, worker in self.detection_workers.items():
            # Unload YOLO model to free computational resources
            worker.unload_model()
            worker.stop()
            worker.join(timeout=5.0)
            logger.info(f"Detection stopped for camera {camera_id}")
        
        self.detection_workers.clear()
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Detection Stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
    
    def _handle_violation(
        self,
        camera_id: int,
        zone_id: int,
        relay_id: int,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        zones_data: List[Tuple[int, List[Tuple[int, int]], int]]
    ) -> None:
        """Handle zone violation.
        
        Args:
            camera_id: Camera identifier
            zone_id: Zone identifier
            relay_id: Relay identifier
            frame: Frame with violation
            person_bbox: Person bounding box (x1, y1, x2, y2)
            zones_data: List of (zone_id, points, relay_id) tuples
        """
        logger.warning(
            f"VIOLATION DETECTED: Camera {camera_id}, Zone {zone_id}, Relay {relay_id}"
        )
        
        # Trigger relay using flexible relay manager
        if relay_id in self.relay_manager.available_channels:
            triggered = self.relay_manager.set_state(
                relay_id,
                True,
                reason=f"Zone {zone_id} violation (Camera {camera_id})"
            )
            
            if triggered:
                logger.info(f"Relay {relay_id} triggered for zone violation")
            else:
                logger.debug(f"Relay {relay_id} in cooldown")
        else:
            logger.warning(
                f"Relay {relay_id} not available. Available: {self.relay_manager.available_channels}"
            )
        
        # Save snapshot
        self._save_snapshot(camera_id, zone_id, relay_id, frame, person_bbox, zones_data)
    
    def _save_snapshot(
        self,
        camera_id: int,
        zone_id: int,
        relay_id: int,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        zones_data: List[Tuple[int, List[Tuple[int, int]], int]]
    ) -> None:
        """Save violation snapshot with zone boundaries and violating person bounding box drawn."""
        try:
            # Create a copy to avoid modifying the original frame
            snapshot_frame = frame.copy()
            
            logger.debug(f"Saving snapshot - Frame shape: {snapshot_frame.shape}, Camera: {camera_id}, Zones count: {len(zones_data)}")
            
            # Draw zone boundaries on the snapshot using zones_data passed from detection worker
            for zone_id_iter, points, relay_id_iter in zones_data:
                logger.debug(f"Processing zone {zone_id_iter}, points: {len(points) if points else 0}")
                
                if not points or len(points) < 2:
                    logger.debug(f"Skipping zone {zone_id_iter} - insufficient points")
                    continue
                
                # Convert points to numpy array for cv2.polylines
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Use bright red (BGR: 0,0,255 = pure red) for violated zone, or default color
                if zone_id_iter == zone_id:
                    # This is the violated zone - highlight in bright red
                    draw_color = (0, 0, 255)  # Pure red in BGR
                    border_width = 4
                    logger.debug(f"Drawing violated zone {zone_id_iter} in red")
                else:
                    # Other zones in green for reference
                    draw_color = (0, 255, 0)  # Green in BGR
                    border_width = 2
                    logger.debug(f"Drawing zone {zone_id_iter} in green")
                
                # Draw polygon outline
                cv2.polylines(snapshot_frame, [pts], True, draw_color, border_width)
                
                # Add zone label
                if len(points) > 0:
                    label = f"Zone {zone_id_iter}"
                    if zone_id_iter == zone_id:
                        label += " [VIOLATION]"
                    cv2.putText(
                        snapshot_frame,
                        label,
                        (int(points[0][0]) + 5, int(points[0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        draw_color,
                        2
                    )
            
            # Draw the violating person's bounding box
            x1, y1, x2, y2 = person_bbox
            logger.debug(f"Drawing person bbox: ({x1}, {y1}, {x2}, {y2})")
            
            # Draw bounding box in pure red (BGR: 0,0,255)
            cv2.rectangle(snapshot_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            # Add label for the violating person
            cv2.putText(
                snapshot_frame,
                "Violating Person",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"violation_cam{camera_id}_zone{zone_id}_relay{relay_id}_{timestamp}.jpg"
            filepath = self.snapshot_dir / filename
            
            cv2.imwrite(str(filepath), snapshot_frame)
            logger.info(f"Snapshot saved with zone boundaries and person bbox: {filename}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}", exc_info=True)
    
    def _update_displays(self) -> None:
        """Update video displays and statistics."""
        for camera_id, video_panel in self.video_panels.items():
            frame = self.camera_manager.get_latest_frame(camera_id)
            if frame is not None:
                video_panel.update_frame(frame)
                
                # Update detected persons
                if camera_id in self.detection_workers:
                    persons = self.detection_workers[camera_id].get_latest_persons()
                    video_panel.set_persons(persons)
                    
                    # Update zone violations (for color change)
                    violations = self.detection_workers[camera_id].get_violations()
                    violation_dict = {zone_id: True for zone_id in violations}
                    video_panel.set_zone_violations(violation_dict)
                
                # Update info
                cap_fps = self.camera_manager.get_fps(camera_id)
                connected = self.camera_manager.is_connected(camera_id)
                
                det_fps = 0.0
                if camera_id in self.detection_workers:
                    det_fps = self.detection_workers[camera_id].get_fps()
                
                status = "Connected" if connected else "Disconnected"
                info = f"Camera {camera_id} | {status} | Cap: {cap_fps:.1f} FPS"
                if self.is_running:
                    info += f" | Det: {det_fps:.1f} FPS"
                
                video_panel.update_info(info)
        
        # Update statistics
        if self.is_running:
            total_zones = sum(
                len(cam.zones)
                for cam in self.config_manager.get_all_cameras()
            )
            self.stats_label.setText(
                f"Monitoring {len(self.video_panels)} cameras, "
                f"{total_zones} restricted zones"
            )
    
    def reload_configuration(self) -> None:
        """Reload configuration after teaching mode changes."""
        if self.is_running:
            self._stop_detection()
        
        # Reload zones
        for camera_id in self.video_panels.keys():
            self._load_zones_for_camera(camera_id)
        
        logger.info("Configuration reloaded")

