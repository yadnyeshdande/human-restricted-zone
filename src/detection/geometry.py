# =============================================================================
# File: detection/geometry.py
# =============================================================================
"""Geometric calculations."""

from typing import Tuple


def point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    """Check if point is inside rectangle.
    
    Args:
        point: (x, y) coordinates
        rect: (x1, y1, x2, y2) rectangle
    
    Returns:
        True if point is inside rectangle
    """
    x, y = point
    x1, y1, x2, y2 = rect
    
    # Ensure proper ordering
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    return x_min <= x <= x_max and y_min <= y <= y_max


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate bounding box center.
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (cx, cy) center coordinates
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)
