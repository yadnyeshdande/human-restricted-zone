#!/usr/bin/env python3
"""
Direct test of USB relay detection without all the application overhead.
"""

import sys
sys.path.insert(0, r'd:\Yadnyesh\Projects\human-restricted-zone\src')

from utils.logger import get_logger

logger = get_logger("RelayTest")

print("\n" + "="*70)
print("DIRECT USB RELAY TEST")
print("="*70 + "\n")

# Step 1: Test basic imports
print("[1] Testing imports...")
try:
    import pyhid_usb_relay
    print("    ✓ pyhid_usb_relay imported")
except Exception as e:
    print(f"    ✗ Failed to import pyhid_usb_relay: {e}")
    sys.exit(1)

# Step 2: Test libusb
print("\n[2] Testing libusb...")
try:
    import libusb
    print(f"    ✓ libusb imported from: {libusb.__file__}")
    
    import pathlib
    dlls = list(pathlib.Path(libusb.__file__).parent.rglob("libusb-1.0.dll"))
    print(f"    ✓ Found {len(dlls)} DLL file(s):")
    for dll in dlls:
        print(f"        - {dll}")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Step 3: Test usb.backend
print("\n[3] Testing usb.backend.libusb1...")
try:
    import usb.backend.libusb1
    print("    ✓ usb.backend.libusb1 imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Step 4: Test getting backend
print("\n[4] Getting backend with patched path...")
try:
    import pathlib
    import usb.backend.libusb1
    import libusb
    
    # Find the x86_64 DLL
    dlls = list(pathlib.Path(libusb.__file__).parent.rglob("x86_64/libusb-1.0.dll"))
    if not dlls:
        dlls = list(pathlib.Path(libusb.__file__).parent.rglob("libusb-1.0.dll"))
    
    if not dlls:
        raise Exception("No libusb-1.0.dll found")
    
    libpath = str(dlls[0])
    print(f"    ✓ DLL found at: {libpath}")
    
    backend = usb.backend.libusb1.get_backend(find_library=lambda x: libpath)
    print(f"    ✓ Backend created: {backend}")
    
except Exception as e:
    import traceback
    print(f"    ✗ Failed: {e}")
    print(f"    Traceback: {traceback.format_exc()}")
    sys.exit(1)

# Step 5: Try to find relay
print("\n[5] Searching for relay device...")
try:
    relay = pyhid_usb_relay.find()
    if relay:
        print(f"    ✓ Relay found: {relay}")
        print(f"    ✓ Relay state: {relay.state}")
        
        # Try to detect channels
        print("\n[6] Detecting relay channels...")
        for ch in range(1, 17):
            try:
                relay.toggle_state(ch)
                print(f"    ✓ Channel {ch} exists")
                relay.toggle_state(ch)  # Toggle back
            except Exception:
                print(f"    ✗ Channel {ch} does NOT exist (last valid: {ch-1})")
                break
        
    else:
        print("    ✗ No relay device found")
        
except Exception as e:
    import traceback
    print(f"    ✗ Failed: {e}")
    print(f"    Traceback:\n{traceback.format_exc()}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")
