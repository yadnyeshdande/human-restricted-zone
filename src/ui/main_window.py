
# =============================================================================
# File: ui/main_window.py
# =============================================================================
"""Main application window."""

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QMessageBox, QAction,
    QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from relay.relay_manager import RelayManager
from .teaching_page import TeachingPage
from .detection_page import DetectionPage
from utils.logger import get_logger
from .settings_page import SettingsPage

logger = get_logger("MainWindow")


class MainWindow(QMainWindow):
    """Main application window with Teaching and Detection modes."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        camera_manager: CameraManager,
        relay_manager: RelayManager
    ):
        """Initialize main window.
        
        Args:
            config_manager: Configuration manager
            camera_manager: Camera manager
            relay_manager: Relay manager
        """
        super().__init__()
        self.config_manager = config_manager
        self.camera_manager = camera_manager
        self.relay_manager = relay_manager
        
        self.setWindowTitle("Industrial Vision Safety System")
        self.setMinimumSize(1280, 720)
        
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        # Central widget with tabs
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # Teaching page
        self.teaching_page = TeachingPage(
            self.config_manager,
            self.camera_manager
        )
        self.teaching_page.zones_changed.connect(self._on_zones_changed)
        self.tabs.addTab(self.teaching_page, "Teaching Mode")
        
        # Detection page
        self.detection_page = DetectionPage(
            self.config_manager,
            self.camera_manager,
            self.relay_manager
        )
        self.tabs.addTab(self.detection_page, "Detection Mode")
        
        # Settings page
        
        self.settings_page = SettingsPage(self.relay_manager)
        self.tabs.addTab(self.settings_page, "Settings")
        
        self.setCentralWidget(self.tabs)

    
    def _setup_menu(self) -> None:
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        save_action = QAction("Save Configuration", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.teaching_page._save_configuration)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        teaching_action = QAction("Teaching Mode", self)
        teaching_action.setShortcut("Ctrl+1")
        teaching_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(teaching_action)
        
        detection_action = QAction("Detection Mode", self)
        detection_action.setShortcut("Ctrl+2")
        detection_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(detection_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        # Shortcuts are already set in menu actions
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+3")
        settings_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        # view_menu.addAction(settings_action)
        pass
    
    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change."""
        if index == 1:  # Detection mode
            # Stop any running detection first
            if self.detection_page.is_running:
                self.detection_page._stop_detection()
            # Reload configuration
            self.detection_page.reload_configuration()
            logger.info("Switched to Detection Mode")
        else:  # Teaching mode
            logger.info("Switched to Teaching Mode")
    
    def _on_zones_changed(self) -> None:
        """Handle zones changed in teaching mode."""
        # Configuration is already updated by teaching page
        logger.debug("Zones configuration changed")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Vision Safety System",
            "<h2>Industrial Vision Safety System</h2>"
            "<p>Version 1.0.0</p>"
            "<p>Multi-camera vision safety monitoring with YOLO detection "
            "and relay-based alerting.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Multi-camera RTSP support</li>"
            "<li>Custom restricted zones</li>"
            "<li>Real-time person detection</li>"
            "<li>Automated relay triggering</li>"
            "<li>24/7 industrial operation</li>"
            "</ul>"
        )
    
    def closeEvent(self, event) -> None:
        """Handle window close with proper cleanup sequence.
        
        CRITICAL: Proper shutdown order to prevent resource leaks and stuck hardware.
        """
        logger.info("=" * 80)
        logger.info("APPLICATION SHUTDOWN INITIATED")
        logger.info("=" * 80)
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?\n\nAll camera connections will be closed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # STEP 1: Stop detection workers (uses cameras and relays)
                logger.info("Step 1: Stopping detection workers...")
                if self.detection_page.is_running:
                    self.detection_page._stop_detection()
                logger.info("  OK: Detection workers stopped")
                
                # STEP 2: Stop UI timers
                logger.info("Step 2: Stopping UI timers...")
                if hasattr(self.detection_page, 'cleanup'):
                    self.detection_page.cleanup()
                if hasattr(self.teaching_page, 'cleanup'):
                    self.teaching_page.cleanup()
                logger.info("  OK: UI timers stopped")
                
                # STEP 3: Shutdown relay manager (CRITICAL - deactivates hardware)
                logger.info("Step 3: Shutting down relay manager...")
                if hasattr(self.relay_manager, 'shutdown'):
                    self.relay_manager.shutdown()
                else:
                    logger.warning("  WARNING: Relay manager has no shutdown method")
                
                # STEP 4: Stop all cameras
                logger.info("Step 4: Stopping cameras...")
                self.camera_manager.shutdown()
                logger.info("  OK: Cameras stopped")
                
                # STEP 5: Save configuration
                logger.info("Step 5: Saving configuration...")
                self.config_manager.save()
                logger.info("  OK: Configuration saved")
                
                logger.info("=" * 80)
                logger.info("APPLICATION SHUTDOWN COMPLETE")
                logger.info("=" * 80)
                
                event.accept()
                
            except Exception as e:
                logger.critical(f"Error during shutdown: {e}", exc_info=True)
                event.accept()  # Accept anyway to prevent hang
        else:
            event.ignore()

