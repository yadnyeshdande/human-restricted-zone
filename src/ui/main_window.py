
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
        """Handle window close."""
        # Stop detection if running
        if self.detection_page.is_running:
            self.detection_page._stop_detection()
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?\n\nAll camera connections will be closed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            logger.info("Application closing...")
            
            # Save configuration
            self.config_manager.save()
            
            # Shutdown camera manager
            self.camera_manager.shutdown()
            
            event.accept()
        else:
            event.ignore()

