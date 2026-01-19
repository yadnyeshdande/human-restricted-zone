# =============================================================================
# File: ui/settings_page.py
# =============================================================================
"""Settings page for application configuration."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QMessageBox, QLineEdit, QScrollArea
)
from PyQt5.QtCore import Qt
from config.app_settings import SETTINGS
from config.config_manager import ConfigManager
from relay.relay_manager import RelayManager
from utils.logger import get_logger

logger = get_logger("SettingsPage")


class SettingsPage(QWidget):
    """Settings configuration interface."""
    
    def __init__(self, relay_manager: RelayManager, config_manager: ConfigManager, parent=None):
        """Initialize settings page.
        
        Args:
            relay_manager: Relay manager instance for testing
            config_manager: Configuration manager for zone rescaling
            parent: Parent widget
        """
        super().__init__(parent)
        self.relay_manager = relay_manager
        self.config_manager = config_manager  # Store for zone rescaling
        self.test_relay_index = 0
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        main_layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Application Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title)
        
        # Scrollable area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        settings_widget = QWidget()
        scroll_layout = QVBoxLayout(settings_widget)
        
        # Processing Settings Group
        processing_group = self._create_processing_group()
        scroll_layout.addWidget(processing_group)
        
        # Detection Settings Group
        detection_group = self._create_detection_group()
        scroll_layout.addWidget(detection_group)
        
        # Relay Settings Group
        relay_group = self._create_relay_group()
        scroll_layout.addWidget(relay_group)
        
        # Performance Settings Group
        performance_group = self._create_performance_group()
        scroll_layout.addWidget(performance_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(settings_widget)
        main_layout.addWidget(scroll)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self._save_settings)
        self.save_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        button_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        button_layout.addWidget(self.status_label)
        
        main_layout.addLayout(button_layout)
    
    def _create_processing_group(self) -> QGroupBox:
        """Create processing settings group."""
        group = QGroupBox("Processing Settings")
        layout = QFormLayout()
        
        # Resolution width
        self.resolution_width = QSpinBox()
        self.resolution_width.setRange(320, 3840)
        self.resolution_width.setSingleStep(160)
        self.resolution_width.setSuffix(" px")
        layout.addRow("Resolution Width:", self.resolution_width)
        
        # Resolution height
        self.resolution_height = QSpinBox()
        self.resolution_height.setRange(240, 2160)
        self.resolution_height.setSingleStep(90)
        self.resolution_height.setSuffix(" px")
        layout.addRow("Resolution Height:", self.resolution_height)
        
        # Quick presets
        preset_layout = QHBoxLayout()
        presets = [
            ("640×360", 640, 360),
            ("960×540", 960, 540),
            ("1280×720", 1280, 720),
            ("1920×1080", 1920, 1080)
        ]
        for name, w, h in presets:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, width=w, height=h: self._set_resolution(width, height))
            preset_layout.addWidget(btn)
        
        layout.addRow("Quick Presets:", preset_layout)
        
        # Info label
        info = QLabel("Lower resolution = Faster processing\nHigher resolution = Better accuracy")
        info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", info)
        
        group.setLayout(layout)
        return group
    
    def _create_detection_group(self) -> QGroupBox:
        """Create detection settings group."""
        group = QGroupBox("Detection Settings")
        layout = QFormLayout()
        
        # YOLO model selection
        self.yolo_model = QComboBox()
        models = [
            ("yolov8n.pt - Fastest, Lowest Accuracy", "yolov8n.pt"),
            ("yolov8s.pt - Fast, Good Accuracy", "yolov8s.pt"),
            ("yolov8m.pt - Medium Speed, Better Accuracy", "yolov8m.pt"),
            ("yolov8l.pt - Slow, High Accuracy", "yolov8l.pt"),
            ("yolov8x.pt - Slowest, Highest Accuracy", "yolov8x.pt")
        ]
        for display, value in models:
            self.yolo_model.addItem(display, value)
        layout.addRow("YOLO Model:", self.yolo_model)
        
        # Confidence threshold
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.1, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        layout.addRow("Confidence Threshold:", self.confidence)
        
        conf_info = QLabel("Lower = More detections (false positives)\nHigher = Fewer detections (may miss people)")
        conf_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", conf_info)
        
        # Violation mode
        self.violation_mode = QComboBox()
        self.violation_mode.addItem("Center - Only when person's center is inside zone", "center")
        self.violation_mode.addItem("Overlap - When person's box overlaps with zone", "overlap")
        layout.addRow("Violation Mode:", self.violation_mode)
        
        mode_info = QLabel(
            "Center = Stricter (person must be inside)\n"
            "Overlap = More sensitive (triggers on any overlap)"
        )
        mode_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", mode_info)
        
        group.setLayout(layout)
        return group
    
    def _create_relay_group(self) -> QGroupBox:
        """Create relay settings group."""
        group = QGroupBox("Relay Settings")
        layout = QFormLayout()
        
        # Use USB relay
        self.use_usb_relay = QCheckBox()
        self.use_usb_relay.stateChanged.connect(self._on_relay_type_changed)
        layout.addRow("Use USB Relay Hardware:", self.use_usb_relay)
        
        # Number of channels
        self.num_channels = QSpinBox()
        self.num_channels.setRange(1, 16)
        layout.addRow("Number of Channels:", self.num_channels)
        
        # Serial number (optional)
        self.usb_serial = QLineEdit()
        self.usb_serial.setPlaceholderText("Leave empty for auto-detect")
        layout.addRow("USB Serial (optional):", self.usb_serial)
        
        # Cooldown
        self.relay_cooldown = QDoubleSpinBox()
        self.relay_cooldown.setRange(0.5, 60.0)
        self.relay_cooldown.setSingleStep(0.5)
        self.relay_cooldown.setSuffix(" seconds")
        layout.addRow("Relay Cooldown:", self.relay_cooldown)
        
        # Duration
        self.relay_duration = QDoubleSpinBox()
        self.relay_duration.setRange(0.1, 10.0)
        self.relay_duration.setSingleStep(0.1)
        self.relay_duration.setSuffix(" seconds")
        layout.addRow("Relay Active Duration:", self.relay_duration)
        
        # Test relay button
        self.test_relay_btn = QPushButton("Test Relays")
        self.test_relay_btn.clicked.connect(self._start_relay_test)
        self.test_relay_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        layout.addRow("", self.test_relay_btn)
        
        relay_info = QLabel(
            "Test will toggle each relay one by one.\n"
            "Click 'Next' to proceed to the next relay."
        )
        relay_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", relay_info)
        
        group.setLayout(layout)
        return group
    
    def _create_performance_group(self) -> QGroupBox:
        """Create performance settings group."""
        group = QGroupBox("Performance Settings")
        layout = QFormLayout()
        
        # Frame queue size
        self.frame_queue_size = QSpinBox()
        self.frame_queue_size.setRange(5, 100)
        self.frame_queue_size.setSingleStep(5)
        layout.addRow("Frame Queue Size:", self.frame_queue_size)
        
        # UI update FPS
        self.ui_fps = QSpinBox()
        self.ui_fps.setRange(10, 60)
        self.ui_fps.setSingleStep(5)
        self.ui_fps.setSuffix(" FPS")
        layout.addRow("UI Update Rate:", self.ui_fps)
        
        perf_info = QLabel(
            "Larger queue = More memory, smoother playback\n"
            "Lower UI FPS = Less CPU usage"
        )
        perf_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", perf_info)
        
        group.setLayout(layout)
        return group
    
    def _set_resolution(self, width: int, height: int) -> None:
        """Set resolution preset."""
        self.resolution_width.setValue(width)
        self.resolution_height.setValue(height)
    
    def _on_relay_type_changed(self, state: int) -> None:
        """Handle relay type change."""
        if state == Qt.Checked:
            self.status_label.setText("USB relay enabled - restart required to apply")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.status_label.setText("Using relay simulator")
            self.status_label.setStyleSheet("color: blue;")
    
    def _load_settings(self) -> None:
        """Load current settings into UI."""
        # Processing
        self.resolution_width.setValue(SETTINGS.processing_resolution[0])
        self.resolution_height.setValue(SETTINGS.processing_resolution[1])
        
        # Detection
        index = self.yolo_model.findData(SETTINGS.yolo_model)
        if index >= 0:
            self.yolo_model.setCurrentIndex(index)
        
        self.confidence.setValue(SETTINGS.detection_confidence)
        
        index = self.violation_mode.findData(SETTINGS.violation_mode)
        if index >= 0:
            self.violation_mode.setCurrentIndex(index)
        
        # Relay
        self.use_usb_relay.setChecked(SETTINGS.use_usb_relay)
        self.num_channels.setValue(SETTINGS.usb_num_channels)
        if SETTINGS.usb_serial:
            self.usb_serial.setText(SETTINGS.usb_serial)
        self.relay_cooldown.setValue(SETTINGS.relay_cooldown)
        self.relay_duration.setValue(SETTINGS.relay_duration)
        
        # Performance
        self.frame_queue_size.setValue(SETTINGS.frame_queue_size)
        self.ui_fps.setValue(SETTINGS.ui_update_fps)
        
        logger.info("Settings loaded into UI")
    
    def _save_settings(self) -> None:
        """Save settings to file."""
        try:
            # Check if resolution changed
            old_resolution = SETTINGS.processing_resolution
            new_resolution = (
                self.resolution_width.value(),
                self.resolution_height.value()
            )
            
            resolution_changed = old_resolution != new_resolution
            
            # Update SETTINGS object
            SETTINGS.processing_resolution = new_resolution
            SETTINGS.yolo_model = self.yolo_model.currentData()
            SETTINGS.detection_confidence = self.confidence.value()
            SETTINGS.violation_mode = self.violation_mode.currentData()
            
            SETTINGS.use_usb_relay = self.use_usb_relay.isChecked()
            SETTINGS.usb_num_channels = self.num_channels.value()
            serial_text = self.usb_serial.text().strip()
            SETTINGS.usb_serial = serial_text if serial_text else None
            SETTINGS.relay_cooldown = self.relay_cooldown.value()
            SETTINGS.relay_duration = self.relay_duration.value()
            
            SETTINGS.frame_queue_size = self.frame_queue_size.value()
            SETTINGS.ui_update_fps = self.ui_fps.value()
            
            # Save to app_settings.json
            SETTINGS.save()
            
            # ========================================
            # SYNC: Update human_boundaries.json if resolution changed
            # ========================================
            if resolution_changed:
                logger.info(f"Resolution changed: {old_resolution} → {new_resolution}")
                self.config_manager.update_processing_resolution(new_resolution)
                self.config_manager.save()
                logger.info("✓ Zones rescaled and saved to human_boundaries.json")
            
            self.status_label.setText("Settings saved successfully!")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            
            # Show appropriate message
            if resolution_changed:
                QMessageBox.information(
                    self,
                    "Settings Saved - Restart Required",
                    "Settings saved successfully!\n\n"
                    "⚠️ IMPORTANT: Please restart the application for these changes to take effect:\n"
                    "- Processing resolution\n"
                    "- YOLO model\n"
                    "- USB relay configuration\n\n"
                    "Zones have been automatically rescaled to the new resolution."
                )
            else:
                QMessageBox.information(
                    self,
                    "Settings Saved",
                    "Settings saved successfully!\n\n"
                    "Note: Some settings require application restart to take effect:\n"
                    "- YOLO model\n"
                    "- USB relay configuration"
                )
            
            logger.info("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            self.status_label.setText("Error saving settings!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")
    
    def _reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to default values?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            SETTINGS.reset_to_defaults()
            self._load_settings()
            self.status_label.setText("Settings reset to defaults")
            self.status_label.setStyleSheet("color: blue;")
            logger.info("Settings reset to defaults")
    
    def _start_relay_test(self) -> None:
        """Start relay testing sequence."""
        self.test_relay_index = 1
        self._test_next_relay()
    
    def _test_next_relay(self) -> None:
        """Test next relay in sequence."""
        max_channels = self.num_channels.value()
        
        if self.test_relay_index > max_channels:
            QMessageBox.information(
                self,
                "Test Complete",
                f"All {max_channels} relays tested successfully!"
            )
            self.status_label.setText("Relay test completed")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            return
        
        # Show dialog for current relay
        msg = QMessageBox(self)
        msg.setWindowTitle(f"Testing Relay {self.test_relay_index}")
        msg.setText(
            f"Testing Relay {self.test_relay_index} of {max_channels}\n\n"
            f"The relay will now toggle ON then OFF.\n"
            f"Please observe the relay indicator/load.\n\n"
            f"Did Relay {self.test_relay_index} activate?"
        )
        msg.setIcon(QMessageBox.Question)
        
        yes_btn = msg.addButton("Yes, it worked", QMessageBox.YesRole)
        no_btn = msg.addButton("No, didn't work", QMessageBox.NoRole)
        next_btn = msg.addButton("Next Relay", QMessageBox.AcceptRole)
        cancel_btn = msg.addButton("Cancel Test", QMessageBox.RejectRole)
        
        # Activate relay
        try:
            logger.info(f"Testing relay {self.test_relay_index}")
            self.relay_manager.trigger(self.test_relay_index)
            self.status_label.setText(f"Testing Relay {self.test_relay_index}...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        except Exception as e:
            logger.error(f"Failed to test relay {self.test_relay_index}: {e}")
            QMessageBox.critical(
                self,
                "Relay Test Error",
                f"Failed to activate relay {self.test_relay_index}:\n{e}"
            )
            return
        
        msg.exec_()
        
        clicked = msg.clickedButton()
        
        if clicked == cancel_btn:
            self.status_label.setText("Relay test cancelled")
            self.status_label.setStyleSheet("color: gray;")
            return
        elif clicked == no_btn:
            QMessageBox.warning(
                self,
                "Relay Issue",
                f"Relay {self.test_relay_index} did not activate.\n\n"
                f"Possible issues:\n"
                f"- Relay hardware not connected\n"
                f"- Incorrect channel number\n"
                f"- Hardware failure\n"
                f"- Check logs for details"
            )
        elif clicked == yes_btn:
            logger.info(f"Relay {self.test_relay_index} test successful")
        
        # Move to next relay
        self.test_relay_index += 1
        self._test_next_relay()