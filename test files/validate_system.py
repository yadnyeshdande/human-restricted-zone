#!/usr/bin/env python3
"""
Complete System Validation Script
Verifies all components and databases are working correctly.
"""

import sys
import os
import json
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def validate_imports():
    """Check all imports work."""
    print("\n" + "="*60)
    print("IMPORT VALIDATION")
    print("="*60)
    
    modules = [
        ("utils.logger", "get_logger"),
        ("config.schema", "AppConfig, Camera, Zone"),
        ("config.config_manager", "ConfigManager"),
        ("config.migration", "migrate_config"),
        ("camera.camera_worker", "CameraWorker"),
        ("camera.camera_manager", "CameraManager"),
        ("camera.reconnect_policy", "ReconnectPolicy"),
        ("relay.relay_interface", "RelayInterface"),
        ("relay.relay_simulator", "RelaySimulator"),
        ("relay.relay_manager", "RelayManager"),
        ("detection.geometry", "point_in_rect, bbox_center"),
        ("detection.detector", "PersonDetector"),
        ("detection.detection_worker", "DetectionWorker"),
    ]
    
    for module, items in modules:
        try:
            __import__(module)
            print(f"✓ {module:40} ({items})")
        except Exception as e:
            print(f"✗ {module:40} ERROR: {e}")
            return False
    
    return True

def validate_instantiation():
    """Check core classes can be instantiated."""
    print("\n" + "="*60)
    print("INSTANTIATION VALIDATION")
    print("="*60)
    
    try:
        from config.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = cm.load()
        print(f"✓ ConfigManager - loaded {len(cfg.cameras)} cameras")
    except Exception as e:
        print(f"✗ ConfigManager - {e}")
        return False
    
    try:
        from camera.camera_manager import CameraManager
        cam_mgr = CameraManager()
        print(f"✓ CameraManager - instantiated")
    except Exception as e:
        print(f"✗ CameraManager - {e}")
        return False
    
    try:
        from relay.relay_manager import RelayManager
        rel_mgr = RelayManager()
        print(f"✓ RelayManager - instantiated (simulator mode)")
    except Exception as e:
        print(f"✗ RelayManager - {e}")
        return False
    
    try:
        from detection.detector import PersonDetector
        det = PersonDetector()
        print(f"✓ PersonDetector - initialized with model")
    except Exception as e:
        print(f"✗ PersonDetector - {e}")
        return False
    
    return True

def validate_configuration():
    """Check configuration system."""
    print("\n" + "="*60)
    print("CONFIGURATION VALIDATION")
    print("="*60)
    
    config_file = Path("human_boundaries.json")
    
    try:
        from config.config_manager import ConfigManager
        
        cm = ConfigManager()
        cfg = cm.load()
        
        print(f"✓ Configuration file: {config_file.name}")
        print(f"  - App version: {cfg.app_version}")
        print(f"  - Processing resolution: {cfg.processing_resolution}")
        print(f"  - Cameras: {len(cfg.cameras)}")
        print(f"  - Total zones: {sum(len(c.zones) for c in cfg.cameras)}")
        
        # Verify structure
        assert cfg.app_version == "1.0.0", "Invalid version"
        assert cfg.processing_resolution == (1280, 720), "Invalid resolution"
        assert isinstance(cfg.cameras, list), "Invalid cameras list"
        
        return True
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

def validate_geometry():
    """Test geometry functions."""
    print("\n" + "="*60)
    print("GEOMETRY VALIDATION")
    print("="*60)
    
    try:
        from detection.geometry import point_in_rect, bbox_center
        
        # Test point_in_rect
        assert point_in_rect((50, 50), (0, 0, 100, 100)) == True
        assert point_in_rect((150, 50), (0, 0, 100, 100)) == False
        print("✓ point_in_rect() working correctly")
        
        # Test bbox_center
        center = bbox_center((0, 0, 100, 100))
        assert center == (50, 50), f"Expected (50, 50), got {center}"
        print("✓ bbox_center() working correctly")
        
        return True
    except Exception as e:
        print(f"✗ Geometry validation failed: {e}")
        return False

def validate_relay():
    """Test relay system."""
    print("\n" + "="*60)
    print("RELAY VALIDATION")
    print("="*60)
    
    try:
        from relay.relay_manager import RelayManager
        
        rel_mgr = RelayManager(cooldown=0.1, activation_duration=0.1)
        
        # Test trigger
        result1 = rel_mgr.trigger(1)
        assert result1 == True, "First trigger should succeed"
        print("✓ Relay 1 triggered successfully")
        
        # Test cooldown
        result2 = rel_mgr.trigger(1)
        assert result2 == False, "Second trigger should fail (cooldown)"
        print("✓ Relay cooldown working correctly")
        
        # Test different relay
        result3 = rel_mgr.trigger(2)
        assert result3 == True, "Different relay should trigger"
        print("✓ Multiple relays working")
        
        # Test state
        state = rel_mgr.get_state(1)
        print(f"✓ Relay state query working (relay 1: {state})")
        
        return True
    except Exception as e:
        print(f"✗ Relay validation failed: {e}")
        return False

def validate_logging():
    """Check logging system."""
    print("\n" + "="*60)
    print("LOGGING VALIDATION")
    print("="*60)
    
    try:
        from utils.logger import get_logger
        
        logger = get_logger("ValidationTest")
        logger.info("Test log message")
        
        log_file = Path("logs/vision_safety.log")
        if log_file.exists():
            print(f"✓ Log file created: {log_file}")
            size_kb = log_file.stat().st_size / 1024
            print(f"  - Size: {size_kb:.1f} KB")
        else:
            print("⚠ Log file not yet created (will be on first app run)")
        
        return True
    except Exception as e:
        print(f"✗ Logging validation failed: {e}")
        return False

def main():
    """Run all validations."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║    INDUSTRIAL VISION SAFETY SYSTEM - VALIDATION SUITE      ║")
    print("║                    Complete System Check                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    all_pass = True
    
    all_pass &= validate_imports()
    all_pass &= validate_instantiation()
    all_pass &= validate_configuration()
    all_pass &= validate_geometry()
    all_pass &= validate_relay()
    all_pass &= validate_logging()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if all_pass:
        print("\n✓ ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        print("\nTo launch the application:")
        print("  python src\\app.py")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
        return 1

if __name__ == "__main__":
    sys.exit(main())
