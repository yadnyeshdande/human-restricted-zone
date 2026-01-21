# =============================================================================
# File: detection/geometry.py
# =============================================================================
"""Geometric calculations."""

from typing import Tuple, List, Optional, Dict

def point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    """Check if point is inside rectangle (kept for backward compatibility).
    
    Args:
        point: (x, y) coordinates
        rect: (x1, y1, x2, y2) rectangle
    
    Returns:
        True if point is inside rectangle
    """
    # Convert rect to polygon and use point_in_polygon
    x1, y1, x2, y2 = rect
    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return point_in_polygon(point, polygon)

# Old version kept for reference
# def point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
#     """Check if point is inside rectangle.
    
#     Args:
#         point: (x, y) coordinates
#         rect: (x1, y1, x2, y2) rectangle
    
#     Returns:
#         True if point is inside rectangle
#     """
#     x, y = point
#     x1, y1, x2, y2 = rect
    
#     # Ensure proper ordering
#     x_min, x_max = min(x1, x2), max(x1, x2)
#     y_min, y_max = min(y1, y2), max(y1, y2)
    
#     return x_min <= x <= x_max and y_min <= y <= y_max


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate bounding box center.
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (cx, cy) center coordinates
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def bbox_overlaps_rect(bbox: Tuple[int, int, int, int], rect: Tuple[int, int, int, int]) -> bool:
    '''Check if bounding box overlaps with rectangle.
    
    Args:
        bbox: (x1, y1, x2, y2) person bounding box
        rect: (x1, y1, x2, y2) restricted zone
    
    Returns:
        True if any overlap exists
    '''
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    rect_x1, rect_y1, rect_x2, rect_y2 = rect
    
    # No overlap if one rectangle is to the left/right/above/below the other
    if bbox_x2 < rect_x1 or bbox_x1 > rect_x2:
        return False
    if bbox_y2 < rect_y1 or bbox_y1 > rect_y2:
        return False
    
    return True

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """Check if point is inside polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinates to test
        polygon: List of (x, y) vertices defining the polygon
    
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def bbox_overlaps_polygon(bbox: Tuple[int, int, int, int], polygon: List[Tuple[int, int]]) -> bool:
    """Check if bounding box overlaps with polygon.
    
    Args:
        bbox: (x1, y1, x2, y2) person bounding box
        polygon: List of (x, y) vertices
    
    Returns:
        True if any overlap exists
    """
    x1, y1, x2, y2 = bbox
    
    # Check if any corner of bbox is inside polygon
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for corner in corners:
        if point_in_polygon(corner, polygon):
            return True
    
    # Check if any polygon vertex is inside bbox
    for px, py in polygon:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    
    # Check if any bbox edge intersects any polygon edge
    bbox_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1))
    ]
    
    n = len(polygon)
    for i in range(n):
        poly_edge = (polygon[i], polygon[(i + 1) % n])
        for bbox_edge in bbox_edges:
            if _segments_intersect(bbox_edge[0], bbox_edge[1], poly_edge[0], poly_edge[1]):
                return True
    
    return False


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Check if line segment p1-p2 intersects with p3-p4."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def polygon_to_rect(polygon: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Get bounding rectangle for polygon.
    
    Args:
        polygon: List of (x, y) points
    
    Returns:
        (x_min, y_min, x_max, y_max) bounding box
    """
    if not polygon:
        return (0, 0, 0, 0)
    
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    return (min(xs), min(ys), max(xs), max(ys))