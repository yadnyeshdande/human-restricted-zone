#!/usr/bin/env python3
"""
Launcher script for the Consolidated Vision Safety System.

This script executes the modular src/app.py application.
"""

import sys
import os

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Add src directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    # Execute the modular application
    from app import main
    sys.exit(main())
