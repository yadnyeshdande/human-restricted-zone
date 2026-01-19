# Quick Start Guide
## Industrial Vision Safety System v1.0.0

### Installation & Setup

#### 1. Prerequisites
- Python 3.10+ (comes with the virtual environment)
- RTSP IP cameras (test format: `rtsp://admin:Pass_123@192.168.1.64:554/stream`)
- GPU with CUDA (optional, CPU will work)

#### 2. Activate Virtual Environment
```powershell
cd "D:\New folder\human"
.\human_env_py310\Scripts\activate
```

#### 3. Run Application
```powershell
python src\app.py
```

The application will open with Teaching Mode active.

---

### Configuration Files

**Location:** `D:\New folder\human\human_boundaries.json`

**Created automatically** with this structure:
```json
{
  "app_version": "1.0.0",
  "timestamp": "2026-01-17T02:55:02",
  "processing_resolution": [1280, 720],
  "cameras": []
}
```

**Logging:** Stored in `logs/vision_safety.log` (rotating, 10MB per file)

---

### Usage Workflow

#### **Teaching Mode** (Define Zones)

1. **Add Camera**
   - Click "Add Camera" button
   - Paste RTSP URL: `rtsp://admin:Pass_123@192.168.1.64:554/stream`
   - Stream loads → Camera panel appears

2. **Draw Zones**
   - Click-and-drag on video to create rectangle
   - Each zone auto-assigns to next relay number

3. **Edit Zones**
   - Click to select zone
   - Drag corners to resize
   - Drag center to move
   - Delete key to remove

4. **Undo/Redo**
   - Ctrl+Z (undo)
   - Ctrl+Y (redo)
   - Up to 50 states saved

5. **Save**
   - Click "Save Configuration"
   - All zones, cameras, relay assignments persist

#### **Detection Mode** (Monitor & Trigger)

1. **Start Detection**
   - Click "Start Detection" button
   - YOLO begins scanning each camera
   - Live FPS displayed per camera

2. **Monitor Violations**
   - Red zones highlight restricted areas
   - Person enters zone → Relay triggers
   - Snapshot auto-saved to `snapshots/` folder
   - Event logged with timestamp

3. **Relay Behavior**
   - Activates for 1 second (configurable)
   - Cooldown: 5 seconds before next trigger
   - Simulator mode (non-blocking)

4. **Stop Detection**
   - Click "Stop Detection" button
   - All workers gracefully shut down

---

### Example Camera URLs

**Generic RTSP IP Camera**
```
rtsp://admin:password@192.168.1.100:554/stream
```

**Hikvision**
```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
```

**Axis**
```
rtsp://admin:password@192.168.1.100:554/axis-media/media.amp
```

**Dahua**
```
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1
```

---

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+1 | Switch to Teaching Mode |
| Ctrl+2 | Switch to Detection Mode |
| Ctrl+3 | Settings |
| Ctrl+S | Save Configuration |
| Ctrl+Z | Undo zone edit |
| Ctrl+Y | Redo zone edit |
| Delete | Remove selected zone |
| Ctrl+Q | Exit application |

---

### Performance Tips

**For Multiple Cameras (4-8):**
- Use nano YOLO model: `yolov8n.pt` (default - fast)
- Processing resolution: 1280×720
- Enable GPU (auto-detected)

**For High Accuracy:**
- Switch to medium model: `yolov8m.pt`
- Increase resolution to 1920×1080
- Requires more compute power

**GPU Memory**
- Nano model: ~250MB
- Small model: ~400MB
- Medium model: ~750MB

---

### Troubleshooting

#### Camera Not Connecting
```
ERROR: Camera 1 connection error: ...
```
- Verify RTSP URL format
- Test URL in VLC player
- Check firewall/network settings
- Ensure camera is powered on

#### Low FPS
- Enable GPU: Check console for "YOLO using GPU (CUDA)"
- Reduce number of cameras
- Lower processing resolution in config
- Close other applications

#### Relay Not Triggering
- Verify zone placement (visual on screen)
- Check cooldown timer (5s minimum between triggers)
- Confirm person detected (check detection FPS)
- Enable YOLO GPU for faster inference

#### No Snapshots
- Check `snapshots/` folder exists (auto-created)
- Ensure write permissions
- Check free disk space

---

### System Requirements

**Minimum:**
- 4GB RAM
- 2-core CPU
- USB or network cameras
- Windows 7+ / Linux / macOS

**Recommended:**
- 8GB RAM
- 4-core CPU
- NVIDIA GPU with CUDA
- Gigabit network

**For 4-8 Cameras:**
- 16GB RAM
- 8-core CPU @ 2.5GHz+
- NVIDIA RTX or better
- SSD for logging

---

### Logging & Monitoring

**Log Location:** `logs/vision_safety.log`

**Log Levels:**
- DEBUG: Frame-by-frame detail
- INFO: System events (cameras added, detection started)
- WARNING: Connection issues, frame drops
- ERROR: Critical failures

**View Live Logs:**
```powershell
Get-Content "logs\vision_safety.log" -Wait
```

---

### Advanced Configuration

Edit `src/config/config_manager.py`:

```python
# Change relay cooldown (seconds)
RelayManager(cooldown=3.0)  # was 5.0

# Change relay duration (seconds)
RelayManager(activation_duration=2.0)  # was 1.0

# Change processing resolution
CameraManager(processing_resolution=(960, 540))  # was (1280, 720)
```

---

### Shutdown

**Graceful Shutdown:**
1. File → Exit
2. Click "Yes" in confirmation dialog
3. All cameras stop
4. Configuration auto-saved
5. Threads join (5-second timeout)

---

### Support

**Configuration File:** `human_boundaries.json`  
**Logs:** `logs/vision_safety.log`  
**Snapshots:** `snapshots/violation_*.jpg`

For issues:
1. Check logs for error messages
2. Verify camera RTSP URL works in VLC
3. Ensure proper Python environment activated
4. Check available disk space (logging, snapshots)

---

**Version:** 1.0.0  
**Status:** Production Ready ✓
