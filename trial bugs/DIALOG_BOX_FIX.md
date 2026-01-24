# Fix: "Starting Detection" Dialog Box Issue

## Problem
When clicking "Start Detection" in detection mode, a dialog box appears briefly with truncated/missing text ("Starting Date...") and disappears immediately.

## Root Cause
The QProgressDialog was created with default minimal dimensions, causing:
1. Text to be cut off or not fully visible
2. Dialog to appear too small and cramped
3. No parent widget specified, causing potential focus/visibility issues

## Solution
Enhanced the progress dialog initialization with:

1. **Added minimum size constraints**
   - `setMinimumWidth(400)`
   - `setMinimumHeight(150)`
   - Ensures dialog is large enough to display full text

2. **Added parent widget reference**
   - `QProgressDialog(..., self)` 
   - Ensures dialog properly inherits from main window

3. **Added explicit repaint**
   - `progress.repaint()` after `.show()`
   - Forces immediate visual update before processing starts

4. **Added camera validation**
   - Checks if cameras are configured before starting
   - Shows helpful message if no cameras exist

## Files Modified
- `src/ui/detection_page.py` (Lines 216-252)

## Code Changes

### Before:
```python
progress = QProgressDialog(
    "Initializing detection system...\n\nLoading YOLO model...",
    None,
    0,
    0
)
progress.setWindowTitle("Starting Detection")
progress.setWindowModality(QtCore.WindowModal)
progress.setCancelButton(None)
progress.show()
```

### After:
```python
progress = QProgressDialog(
    "Initializing detection system...\n\nLoading YOLO model...",
    None,
    0,
    0,
    self  # ← Added parent widget
)
progress.setWindowTitle("Starting Detection")
progress.setWindowModality(QtCore.WindowModal)
progress.setCancelButton(None)
progress.setMinimumWidth(400)   # ← Added min width
progress.setMinimumHeight(150)  # ← Added min height
progress.show()

# Force immediate repaint to ensure dialog is visible
progress.repaint()  # ← Added repaint
```

## Testing
1. Go to Detection Mode
2. Click "Start Detection"
3. Dialog should now:
   - Display full "Initializing detection system..." text
   - Stay visible for the full initialization process
   - Show model loading progress
   - Close automatically when detection starts

## Additional Safety
Added check for configured cameras before starting detection:
```python
if not cameras:
    progress.close()
    QMessageBox.warning(self, "No Cameras", "No cameras configured. Please add cameras first.")
    return
```

This prevents unnecessary processing if no cameras are set up.
