#!/usr/bin/env python3
"""
Launcher script for the Consolidated Vision Safety System.

This script executes the consolidated human_onefile_ui_remaining.py file
and handles any startup errors gracefully.
"""

import sys
import os

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Execute the consolidated application
    import human_onefile_polygon_ui_remaining
    human_onefile_polygon_ui_remaining.main()
