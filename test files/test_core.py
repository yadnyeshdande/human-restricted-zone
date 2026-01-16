#!/usr/bin/env python3
"""Quick test script to verify core functionality."""

import sys
import os

# Set up paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test imports
print("Testing core imports...")
try:
    from config.config_manager import ConfigManager
    from camera.camera_manager import CameraManager
    from relay.relay_manager import RelayManager
    from detection.detector import PersonDetector
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test instantiation
print("\nTesting instantiation...")
try:
    config_manager = ConfigManager()
    config = config_manager.load()
    print(f"✓ ConfigManager: loaded {len(config.cameras)} cameras")
    
    camera_manager = CameraManager()
    print(f"✓ CameraManager: instantiated")
    
    relay_manager = RelayManager()
    print(f"✓ RelayManager: instantiated (using simulator)")
    
    print("\n✓ All core tests passed!")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
