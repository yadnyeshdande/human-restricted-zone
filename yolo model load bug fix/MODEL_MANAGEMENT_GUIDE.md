# YOLO Model Management Guide

## Overview
This document explains how YOLO models are loaded, where feedback is provided, and how the path system works.

---

## 1. When & Where YOLO Model Loads

### Initial Load (Application Startup)

**File:** `src/app.py` (lines 34-58)

When you start the application:
```
2026-01-22 04:08:28 - VisionSafety.Main - INFO - Loading application settings...
2026-01-22 04:08:28 - VisionSafety.Main - INFO - ✓ YOLO Model found: yolov8n.pt (6.43 MB)
```

**Feedback shown:**
- ✓ If model exists: Shows model name and file size
- ⚠️ If model missing: Shows warning with clear instructions on what to do

---

### Loading During Detection Start

**File:** `src/ui/detection_page.py` (lines 215-280)

When you click "▶ Start Detection":

1. **Progress Dialog appears** with message:
   ```
   Initializing detection system...
   Loading YOLO model for camera 1...
   ```

2. **Model loads** (takes 5-15 seconds depending on model size):
   - Small models (yolov8n): ~5-10 seconds
   - Large models (yolov8x): ~15-30 seconds

3. **Feedback shown:**
   - ✓ If successful: Green status "✓ Detection Running" + logs confirmation
   - ❌ If failed: Error dialog with instructions to download model from Settings

**Logs shown during load:**
```
2026-01-22 04:10:15 - VisionSafety.Detector - INFO - Initializing YOLO detector: yolov8n.pt
2026-01-22 04:10:15 - VisionSafety.Detector - INFO - ✓ Model found in local storage: d:\New folder\human\models\yolov8n.pt
2026-01-22 04:10:15 - VisionSafety.Detector - INFO -   Model size: 6.43 MB
2026-01-22 04:10:15 - VisionSafety.Detector - INFO - Loading model: yolov8n.pt...
2026-01-22 04:10:21 - VisionSafety.Detector - INFO - ✓ Model loaded successfully: yolov8n.pt
2026-01-22 04:10:21 - VisionSafety.Detector - INFO - ✓ YOLO using CPU
```

---

## 2. User Feedback Mechanisms

### In Settings → Detection Settings

**Button:** "📥 Check & Download"

**Actions:**
1. Click button → Shows model status dialog
2. If model not found → Offers to download
3. Download starts → Progress dialog shows "Downloading model..."
4. After download → Success confirmation with file location

**Status Display:**
```
Status: ✓ Loaded (6.43 MB)        ← Model is ready
Status: ⚠️ Not found - Click 'Check & Download' to download  ← Model missing
```

---

### During Detection Start

**Progress Dialog:**
```
Initializing detection system...
Loading YOLO model for camera 1...
```

**After successful load:**
- Status changes to: **"✓ Detection Running"** (Green)
- Logs show: `✓ Detection started for camera 1`

**If model fails to load:**
- Error dialog appears with instructions
- User directed to Settings → Detection Settings
- "Check & Download" button to fix issue

---

### In Application Logs

**File location:** `logs/vision_safety_*.log`

**Entry points show model status:**

1. **App startup:**
   ```
   INFO - ✓ YOLO Model found: yolov8n.pt (6.43 MB)
   ```

2. **Detection worker creation:**
   ```
   INFO - ✓ Detection worker initialized for camera 1
   ```

3. **Model initialization:**
   ```
   INFO - ✓ Model loaded successfully: yolov8n.pt
   INFO - ✓ YOLO using GPU (CUDA)
   ```

**If problems occur:**
```
ERROR - ⚠️ CRITICAL: Model failed to load for camera 1
ERROR - ⚠️ CRITICAL: Cannot start detection without model
WARNING - Model not found in: d:\New folder\human\models
```

---

## 3. Path System (NOT Hardcoded)

### Directory Structure

```
d:\New folder\human\
├── models/                          ← Models stored here (you choose this location)
│   ├── yolov8n.pt                  ← Model file (6.4 MB)
│   ├── yolov8m.pt                  ← Model file (49 MB)
│   └── yolov8l.pt                  ← Model file (82 MB)
├── src/
│   ├── app.py                       ← Uses relative path
│   ├── detection/
│   │   └── detector.py              ← Uses relative path
│   └── ui/
│       └── settings_page.py         ← Uses relative path
├── app_settings.json
└── human_boundaries.json
```

### How Relative Paths Work

**In detector.py:**
```python
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
```

**Breaks down to:**
- `__file__` = `d:\New folder\human\src\detection\detector.py`
- `.parent` = `d:\New folder\human\src\detection`
- `.parent` = `d:\New folder\human\src`
- `.parent` = `d:\New folder\human`
- `/ "models"` = `d:\New folder\human\models`

### Why This is Better (NOT Hardcoded)

✅ **Advantages:**
- Works from ANY directory you run the app from
- Works if you move the entire folder to a different location
- Works on different computers with different paths
- Works on Windows, Mac, Linux with forward slashes automatically converted

❌ **Hardcoded would be:**
```python
MODELS_DIR = r"D:\New folder\human\models"  # ❌ BAD - only works on this specific computer!
```

### Same Pattern Used Everywhere

**File:** `src/ui/settings_page.py`
```python
models_dir = Path(__file__).parent.parent.parent / "models"
```

**File:** `src/ui/detection_page.py`
```python
snapshot_dir = Path("snapshots")  # Relative to where app runs from
```

**File:** `src/app.py`
```python
models_dir = Path(__file__).parent.parent / "models"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

---

## 4. Complete Model Loading Flow

```
┌─────────────────────────────────────────┐
│ User starts application                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ app.py loads settings                   │
│ Checks: Does models/yolov8n.pt exist?   │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   YES ✓            NO ⚠️
   │                │
   │                ▼
   │          Show warning in logs:
   │          "Model not found"
   │          "Use Settings to download"
   │                │
   └─────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ Application fully loaded                 │
│ User can work in Teaching Mode           │
│ (no model needed yet)                    │
└──────────────┬──────────────────────────┘
               │
        User clicks START DETECTION
               │
               ▼
┌─────────────────────────────────────────┐
│ Progress dialog appears:                │
│ "Loading YOLO model..."                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ detector.py creates PersonDetector      │
│ Loads model from: models/yolov8n.pt     │
│ Loading takes 5-15 seconds              │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   SUCCESS ✓       FAIL ❌
   │                │
   │                ▼
   │          Error dialog:
   │          "Model load failed"
   │          "Click Settings to download"
   │                │
   │                ▼
   │          Back to setup
   │
   ▼
┌─────────────────────────────────────────┐
│ Status: "✓ Detection Running" (Green)   │
│ Detection worker processes frames       │
│ Model detects people in real-time       │
└─────────────────────────────────────────┘
```

---

## 5. Troubleshooting

### Problem: "Status: ⚠️ Not found"

**Solution:**
1. Go to Settings → Detection Settings
2. Click "📥 Check & Download" button
3. Select desired model (wait for download to complete)
4. Click "Save Settings"
5. Restart application

### Problem: "Model failed to load"

**Check:**
1. Is file in `models/` folder? 
   ```
   d:\New folder\human\models\yolov8n.pt
   ```
2. Is file not corrupted? (Check size in Settings)
3. Do you have enough RAM? (Models need 2-4 GB)

### Problem: "UnboundLocalError: Path referenced before assignment"

**Fix:** This has been corrected in the latest version. Ensure you have:
```python
from pathlib import Path  # At top of file
```

### Problem: Download takes too long or fails

**Possible causes:**
- Slow internet (models are 30-200 MB)
- Disk space issue
- Firewall blocking download

**Solution:**
- Check internet connection
- Ensure 500 MB free disk space
- Download model manually from: https://github.com/ultralytics/assets/releases

---

## 6. Summary

| Aspect | Details |
|--------|---------|
| **Load Time** | 5-15 seconds per model (first time) |
| **User Feedback** | Logs + Progress dialogs + Status messages |
| **Path Type** | Relative (uses `__file__` and Path class) |
| **Model Location** | `d:\New folder\human\models\` |
| **Download UI** | Settings → Detection Settings → "📥 Check & Download" |
| **Status Display** | Settings page shows ✓ or ⚠️ icon |
| **Error Handling** | Dialog boxes guide user to fix issues |
| **Logs Location** | `logs/vision_safety_*.log` |

