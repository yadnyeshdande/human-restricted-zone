# =============================================================================
# File: app.py - APPLICATION ENTRY POINT
# =============================================================================

import sys
import os

# Add src to path if running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from utils.logger import get_logger
from config.config_manager import ConfigManager
from camera.camera_manager import CameraManager
from relay.relay_manager import RelayManager
from config.app_settings import SETTINGS  # ← IMPORT AT TOP!

logger = get_logger("Main")


def main():
    """Application entry point."""
    logger.info("=" * 80)
    logger.info("INDUSTRIAL VISION SAFETY SYSTEM - STARTING")
    logger.info("=" * 80)
    
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Vision Safety System")
        app.setStyle("Fusion")
        
        # ========================================
        # CRITICAL: Load app settings FIRST
        # ========================================
        logger.info("Loading application settings...")
        SETTINGS.load()  # This loads app_settings.json
        
        # ========================================
        # Initialize configuration manager
        # ========================================
        logger.info("Initializing configuration manager...")
        config_manager = ConfigManager()
        config = config_manager.load()  # This loads human_boundaries.json
        
        # ========================================
        # SYNC CHECK: Detect resolution mismatch
        # ========================================
        app_resolution = SETTINGS.processing_resolution
        config_resolution = config.processing_resolution
        
        if app_resolution != config_resolution:
            logger.warning(f"Resolution mismatch detected!")
            logger.warning(f"  App settings: {app_resolution}")
            logger.warning(f"  Config file:  {config_resolution}")
            logger.info(f"Rescaling zones to match app settings...")
            
            # Update config to match app settings and rescale zones
            config_manager.update_processing_resolution(app_resolution)
            config_manager.save()
            
            logger.info(f"✓ Zones rescaled and saved")
        else:
            logger.info(f"✓ Resolution in sync: {app_resolution}")
        
        # ========================================
        # Initialize camera manager with SYNCED resolution
        # ========================================
        logger.info("Initializing camera manager...")
        camera_manager = CameraManager(
            processing_resolution=SETTINGS.processing_resolution  # Use app settings!
        )
        
        # ========================================
        # Initialize relay manager
        # ========================================
        logger.info("Initializing relay manager...")
        
        # Choose relay interface based on settings
        relay_interface = None
        if SETTINGS.use_usb_relay:
            try:
                from relay.relay_usb_hid import RelayUSBHID
                relay_interface = RelayUSBHID(
                    num_channels=SETTINGS.usb_num_channels,
                    serial=SETTINGS.usb_serial
                )
                logger.info("Using pyhid_usb_relay hardware")
            except Exception as e:
                logger.error(f"Failed to initialize USB relay: {e}")
                logger.error("Falling back to relay simulator")
        
        relay_manager = RelayManager(
            interface=relay_interface,
            cooldown=SETTINGS.relay_cooldown,
            activation_duration=SETTINGS.relay_duration
        )
        
        # ========================================
        # Start cameras from configuration
        # ========================================
        for camera in config.cameras:
            logger.info(f"Starting camera {camera.id}: {camera.rtsp_url}")
            camera_manager.add_camera(camera.id, camera.rtsp_url)
        
        # ========================================
        # Create and show main window
        # ========================================
        logger.info("Creating main window...")
        window = MainWindow(config_manager, camera_manager, relay_manager)
        window.show()
        
        logger.info("Application initialized successfully")
        logger.info("=" * 80)
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()



# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================
"""
This is a production-ready foundation for the multi-camera industrial vision
safety system. The complete implementation requires:

COMPLETED MODULES:
✓ utils/ - Logging, threading, time utilities
✓ config/ - Configuration management and persistence
✓ detection/ - YOLO detector, geometry, detection workers
✓ camera/ - RTSP capture, reconnection policy, camera management
✓ relay/ - Hardware abstraction, simulation, relay management

REMAINING MODULES (PyQt5 UI):
□ ui/main_window.py - Main application window with tabs
□ ui/teaching_page.py - Zone editor interface
□ ui/detection_page.py - Live detection display
□ ui/video_panel.py - Individual camera display widget
□ ui/zone_editor.py - Zone drawing and editing logic

KEY FEATURES IMPLEMENTED:
✓ Thread-safe multi-camera support
✓ RTSP auto-reconnect with exponential backoff
✓ YOLO person detection (GPU/CPU)
✓ Geometric zone violation detection
✓ Sequential relay assignment
✓ Configuration persistence (human_boundaries.json)
✓ Rotating logs
✓ FPS monitoring
✓ Graceful shutdown
✓ Fail-safe error handling

PRODUCTION DEPLOYMENT:
1. Install dependencies: pip install PyQt5 opencv-python ultralytics numpy
2. Create directory structure as shown
3. Place each module in correct subdirectory
4. Implement remaining UI modules (PyQt5)
5. Test with actual RTSP cameras
6. Configure relay hardware interface (replace simulator)
7. Deploy on industrial PC

TESTING:
- Use rtsp://admin:Pass_123@192.168.1.64:554/stream format
- Test with multiple cameras
- Verify relay triggering
- Check log files in logs/
- Validate human_boundaries.json structure

SECURITY NOTES:
- RTSP credentials stored in plaintext (as per spec)
- For production, consider encryption
- Implement access controls
- Secure network configuration

This foundation provides all core business logic, thread management,
detection pipeline, and persistence. The UI layer (PyQt5) would connect
these components with video display, zone drawing, and controls.
"""




# =============================================================================
# File: app.py - UPDATED APPLICATION ENTRY POINT
# =============================================================================
"""
Main application entry point - COMPLETE VERSION

Usage:
    python app.py

Requirements:
    pip install PyQt5 opencv-python ultralytics numpy

Directory Structure:
    src/
    ├── app.py
    ├── camera/
    │   ├── __init__.py
    │   ├── camera_worker.py
    │   ├── camera_manager.py
    │   └── reconnect_policy.py
    ├── detection/
    │   ├── __init__.py
    │   ├── detector.py
    │   ├── detection_worker.py
    │   └── geometry.py
    ├── ui/
    │   ├── __init__.py
    │   ├── main_window.py
    │   ├── teaching_page.py
    │   ├── detection_page.py
    │   ├── video_panel.py
    │   └── zone_editor.py
    ├── relay/
    │   ├── __init__.py
    │   ├── relay_manager.py
    │   ├── relay_interface.py
    │   └── relay_simulator.py
    ├── config/
    │   ├── __init__.py
    │   ├── schema.py
    │   ├── config_manager.py
    │   └── migration.py
    └── utils/
        ├── __init__.py
        ├── logger.py
        ├── threading.py
        └── time_utils.py
"""

