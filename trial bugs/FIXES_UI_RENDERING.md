# UI Rendering Fixes - Zone Offset and Detection Page Blank

## Issues Fixed

### 1. **Zone Offset Duplication Issue** ✓
**Problem:** When drawing a zone in teaching mode, it creates a duplicate zone at an offset position.

**Root Cause:** The `_update_zone_visuals()` method was being called inside the timer loop in `_update_frames()`. This method redraws all zones on the video panel. By calling it every frame while also updating geometry, it was causing the zone editor to render zones twice - once from the stored geometry and once from the frame update.

**Fix Applied:**
- Removed the `self._update_zone_visuals(camera_id)` call from the geometry update section
- Geometry is now updated only when needed (when size changes)
- Zone visuals are only updated when explicitly triggered (zone creation, deletion, or manual edit)

**File:** `src/ui/teaching_page.py` (Line 510)

**Code Change:**
```python
# BEFORE (causing offset):
if zone_editor.geometry() != target_rect:
    zone_editor.setGeometry(target_rect)
    self._update_zone_visuals(camera_id)  # ← REMOVED THIS

# AFTER (fixed):
if zone_editor.geometry() != target_rect:
    zone_editor.setGeometry(target_rect)
```

---

### 2. **Detection Page Blank Screen** ✓
**Problem:** Detection mode shows a completely blank screen with no video, though detection is working in the background.

**Root Cause:** Multiple contributing factors:
1. Video widget size could be zero or uninitialized when frame rendering starts
2. No null-check for empty or invalid frames
3. Scaling calculation could fail if widget size is invalid

**Fixes Applied:**

#### In `src/ui/video_panel.py`:
- Added null/empty frame validation in `update_frame()`
- Added fallback for uninitialized widget size
- Added safety check in scaling calculation to avoid division by zero

**Code Changes:**
```python
# 1. Added frame validation:
def update_frame(self, frame: np.ndarray) -> None:
    if frame is None or frame.size == 0:
        return
    self.current_frame = frame
    self._render_frame()

# 2. Added widget size fallback in _render_frame():
if widget_size.width() <= 0 or widget_size.height() <= 0:
    widget_size.setWidth(max(640, self.processing_width))
    widget_size.setHeight(max(360, self.processing_height))

# 3. Added safe division:
self.scale = scaled_pixmap.width() / width if width > 0 else 1.0
```

---

### 3. **Additional Unicode Encoding Errors** ✓
**Problem:** Arrow character (→) and checkmark character (✓) cause console encoding errors on Windows.

**Root Cause:** Windows console uses cp1252 encoding which doesn't support these Unicode characters.

**Fixes Applied:**
- Replaced arrow (→) with dash arrow (->) in settings_page.py
- Replaced checkmark (✓) with [OK] text in settings_page.py

**Files Modified:** `src/ui/settings_page.py`
- Line 362: `{old_resolution} → {new_resolution}` → `{old_resolution} -> {new_resolution}`
- Line 365: `✓ Zones rescaled` → `[OK] Zones rescaled`

---

## Testing Checklist

### Zone Drawing (Teaching Mode)
- [ ] Draw a zone - should appear only ONCE (no offset copy)
- [ ] Draw multiple zones - each should appear once in correct position
- [ ] Resize window - zones should rescale correctly
- [ ] Delete zone - offset zone should also be removed
- [ ] Save configuration - zones should persist correctly

### Detection Mode
- [ ] Switch to detection mode - should show live video from camera
- [ ] Video should display in each camera panel
- [ ] Zones should be drawn as overlay on the video
- [ ] Start detection - detection should work and show in logs
- [ ] Multiple cameras - each panel should show video independently

### General
- [ ] No Unicode encoding errors in console
- [ ] Settings page shows resolution changes without errors
- [ ] Application runs without crashes

---

## Technical Details

### Zone Rendering Flow (After Fix)

1. **Timer Loop (_update_frames):**
   - Gets latest frame from camera
   - Calls `video_panel.update_frame(frame)`
   - Updates zone editor geometry IF changed (NOT visuals)

2. **Zone Visual Update (Only when triggered):**
   - Zone creation → `_update_zone_visuals()`
   - Zone deletion → `_update_zone_visuals()`
   - Zone edit → `_update_zone_visuals()`
   - NOT called from timer loop

3. **VideoPanel Rendering:**
   - `update_frame()` converts BGR to RGB
   - Draws zones from `self.zones` list
   - Scales to fit widget
   - Updates display label

### Frame Rendering Flow (Detection Mode - After Fix)

1. **Timer loop calls `_update_displays()`**
2. **Gets frame from camera_manager** 
   - `frame = self.camera_manager.get_latest_frame(camera_id)`
3. **Validates frame is not None/empty** (NEW)
   - Added null check
4. **Calls `video_panel.update_frame(frame)`**
5. **VideoPanel validates widget size** (NEW)
   - Falls back to default size if uninitialized
6. **Renders frame with zone overlay**
7. **Scales to fit widget with safety checks** (NEW)
   - Added divide-by-zero protection
8. **Updates display label**

---

## Related Files

- `src/ui/teaching_page.py` - Zone rendering in teaching mode
- `src/ui/video_panel.py` - Frame display and zone overlay
- `src/ui/detection_page.py` - Detection mode UI
- `src/ui/settings_page.py` - Settings with Unicode fixes

---

## Prevention Notes

1. **Avoid repetitive visual updates in timer loops** - Only update when needed
2. **Always validate widget size before rendering** - Especially in newly created widgets
3. **Use ASCII characters in logging** - Avoid Unicode special characters for Windows compatibility
4. **Add defensive null checks** - Especially for frame/image data

