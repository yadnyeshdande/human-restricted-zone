# YOLO Model Selection and Configuration Guide

## How the Application Works (Overview)

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION FLOW                             │
└─────────────────────────────────────────────────────────────────┘

1. START APPLICATION
   └─> python src/app.py

2. INITIALIZE MANAGERS
   ├─> ConfigManager       (Load human_boundaries.json)
   ├─> CameraManager       (Manage RTSP streams)
   └─> RelayManager        (Control relay outputs)

3. CREATE UI
   └─> MainWindow (PyQt5)
       ├─> Teaching Mode   (Add cameras, draw zones)
       └─> Detection Mode  (Live monitoring)

4. DETECTION PIPELINE (Per Camera)
   ├─> CameraWorker       (Thread 1: Capture RTSP frames)
   ├─> DetectionWorker    (Thread 2: Run YOLO inference)
   │   ├─> PersonDetector (YOLO model)
   │   ├─> Geometry Check (Zone boundaries)
   │   └─> Relay Trigger  (Activate relay)
   └─> VideoPanel         (Display video + zones)
```

## How YOLO Model Selection Works

### Current Default (yolov8n.pt)

**File:** `src/detection/detector.py` (Line 18-19)

```python
def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.5):
    """Initialize detector.
    
    Args:
        model_name: YOLO model name
        conf_threshold: Confidence threshold
    """
```

**How it works:**

1. **Default Parameter**: When `PersonDetector()` is instantiated without arguments, it uses `'yolov8n.pt'` (YOLOv8 **nano** - fastest, lowest accuracy)

2. **Model Download**: First time use, ultralytics automatically downloads the model (~6 MB)

3. **GPU Detection** (Lines 30-41):
```python
try:
    from ultralytics import YOLO
    self.model = YOLO(model_name)
    
    # Try to use CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info("YOLO using GPU (CUDA)")
        else:
            logger.info("YOLO using CPU")
    except ImportError:
        logger.info("YOLO using CPU (PyTorch not available)")
```

4. **Detection**: Runs inference with `detect_persons()` method

---

## YOLO Model Options

### YOLOv8 Variants (Speed vs Accuracy Tradeoff)

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy | Best For |
|-------|------|-----------|-----------|----------|----------|
| **yolov8n.pt** | 6 MB | ~200ms | ~15ms | Lower | Real-time, limited resources |
| **yolov8s.pt** | 22 MB | ~100ms | ~8ms | Medium | Balanced |
| **yolov8m.pt** | 49 MB | ~80ms | ~5ms | **Higher** | Production accuracy |
| **yolov8l.pt** | 104 MB | ~60ms | ~3ms | Very High | High-accuracy systems |
| **yolov8x.pt** | 202 MB | ~45ms | ~2ms | Highest | Maximum accuracy |

**Your Question**: To switch to **yolov8m.pt** for higher accuracy, read below.

---

## How to Switch YOLO Models

### Option 1: Edit detector.py (Permanent Change)

**File:** `src/detection/detector.py` Line 18

**Current:**
```python
def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.5):
```

**Change to (yolov8m for better accuracy):**
```python
def __init__(self, model_name: str = 'yolov8m.pt', conf_threshold: float = 0.5):
```

**Impact:**
- All detection workers will use yolov8m.pt
- First run will download ~49 MB model
- Detection will be more accurate but slower (~5-8ms per frame on GPU)

### Option 2: Use Configuration File (Flexible)

Add to `human_boundaries.json`:

```json
{
  "version": "1.0",
  "processing_resolution": [1280, 720],
  "yolo_model": "yolov8m.pt",
  "confidence_threshold": 0.5,
  "cameras": [...],
  "zones": [...]
}
```

Then modify `detection_worker.py` Line 43 to read from config:

```python
try:
    # Load from config instead of hardcoded
    model_name = 'yolov8m.pt'  # Or read from config
    self.detector = PersonDetector(model_name=model_name)
    logger.info(f"Detection worker initialized for camera {camera_id}")
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    raise
```

### Option 3: Dynamic Selection at Runtime (Most Flexible)

**Modify detector.py to allow runtime config:**

```python
class PersonDetector:
    """YOLO-based person detector."""
    
    PERSON_CLASS_ID = 0
    
    # Class variable for model selection
    _model_name = 'yolov8n.pt'  # Can be changed at runtime
    
    def __init__(self, model_name: Optional[str] = None, conf_threshold: float = 0.5):
        """Initialize detector.
        
        Args:
            model_name: YOLO model name (if None, uses class default)
            conf_threshold: Confidence threshold
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.device = 'cpu'
        
        # Use provided model or class default
        actual_model = model_name or PersonDetector._model_name
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(actual_model)
            logger.info(f"Loaded YOLO model: {actual_model}")
            
            # Try to use CUDA if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.info("YOLO using GPU (CUDA)")
                else:
                    logger.info("YOLO using CPU")
            except ImportError:
                logger.info("YOLO using CPU (PyTorch not available)")
                
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            raise
    
    @classmethod
    def set_default_model(cls, model_name: str):
        """Change default model for all future instances."""
        cls._model_name = model_name
        logger.info(f"Default YOLO model set to: {model_name}")
```

Then in `app.py`, before creating detection workers:

```python
from detection.detector import PersonDetector

# Switch to yolov8m for higher accuracy
PersonDetector.set_default_model('yolov8m.pt')

# Now all detection workers will use yolov8m
logger.info("Switching to yolov8m.pt for higher accuracy")
```

---

## Complete Workflow Example: Using yolov8m.pt

### Step 1: Verify Your GPU
```python
# Run this in a terminal to check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Update detector.py
Change line 18 from:
```python
def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.5):
```

To:
```python
def __init__(self, model_name: str = 'yolov8m.pt', conf_threshold: float = 0.5):
```

### Step 3: Launch Application
```bash
python src/app.py
```

First run will download yolov8m.pt (~49 MB) - this takes 1-2 minutes.

### Step 4: Check Logs
```
INFO - Loaded YOLO model: yolov8m.pt
INFO - YOLO using GPU (CUDA)
```

---

## Performance Comparison

### With GPU (NVIDIA CUDA)
```
yolov8n.pt: 15-30ms per frame → ~33-67 FPS
yolov8m.pt:  5-10ms per frame → ~100-200 FPS
yolov8l.pt:  3-5ms per frame  → ~200-333 FPS
```

### With CPU (No GPU)
```
yolov8n.pt: 200-400ms per frame → ~2-5 FPS
yolov8m.pt: 500-800ms per frame → ~1-2 FPS (NOT RECOMMENDED)
yolov8l.pt: 1000-2000ms per frame → Not usable
```

**Recommendation**: 
- **GPU available**: Use yolov8m.pt or yolov8l.pt for accuracy
- **CPU only**: Stick with yolov8n.pt for real-time performance

---

## Confidence Threshold Settings

In `detector.py` line 19:
```python
def __init__(self, model_name: str = 'yolov8m.pt', conf_threshold: float = 0.5):
```

**conf_threshold** affects detection sensitivity:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.3** | Very sensitive, many false positives | Maximum coverage |
| **0.5** | Balanced (default) | Standard security |
| **0.7** | Strict, fewer false positives | High precision required |
| **0.9** | Very strict, may miss detections | Minimal false alarms |

**Example**: Change to stricter threshold:
```python
self.detector = PersonDetector(model_name='yolov8m.pt', conf_threshold=0.7)
```

---

## Quick Reference: Complete Setup for yolov8m.pt

### File: `src/detection/detector.py`
```python
def __init__(self, model_name: str = 'yolov8m.pt', conf_threshold: float = 0.65):
    """Initialize detector with yolov8m for higher accuracy."""
    # ... rest of code unchanged
```

Then start your app:
```bash
python src/app.py
```

**Expected Output**:
```
INDUSTRIAL VISION SAFETY SYSTEM - STARTING
Initializing configuration manager...
Initializing camera manager...
Initializing relay manager...
Creating main window...
Loaded YOLO model: yolov8m.pt
YOLO using GPU (CUDA)
Application initialized successfully
```

---

## Troubleshooting

### Issue: Model not found/downloading slow
```
Solution: Models cache in ~/.yolov8/ 
First run takes 1-2 min, subsequent runs load from cache
```

### Issue: GPU not detected
```
Solution: Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

If False, install CUDA toolkit or use CPU with yolov8n.pt
```

### Issue: Out of memory on GPU
```
Solution: Switch to smaller model (yolov8n.pt) or reduce batch size
PersonDetector.set_default_model('yolov8n.pt')
```

### Issue: Too slow on CPU
```
Solution: Use yolov8n.pt (default nano model)
Or enable GPU: Install CUDA 11.8+ and PyTorch with CUDA support
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Default Model** | yolov8n.pt (nano - fastest) |
| **For High Accuracy** | yolov8m.pt (medium - balanced) |
| **For Maximum Accuracy** | yolov8l.pt (large) or yolov8x.pt (xlarge) |
| **Change Method** | Edit `detector.py` line 18-19 |
| **GPU Required For** | Real-time processing with medium/large models |
| **CPU Performance** | Only nano model (yolov8n.pt) is usable |

**To switch to yolov8m.pt**: Simply change one line in `detector.py` and restart!
