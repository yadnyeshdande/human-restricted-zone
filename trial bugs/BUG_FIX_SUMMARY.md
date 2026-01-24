# Bug Fix Summary - Camera Connection Issues

## Issues Identified and Fixed

### 1. **Unicode Encoding Error (✓ checkmark character)**
**Problem:** Windows console uses cp1252 encoding which doesn't support the checkmark character (✓). This caused logging errors when trying to print status messages.

**Symptoms:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

**Root Cause:** Using Unicode special characters in log messages on Windows with cp1252 encoding.

**Fix Applied:** Replaced checkmark character with `[OK]` text in:
- `src/app.py` line 89 and 93

**Files Modified:**
- `src/app.py`

---

### 2. **Missing QRect Import**
**Problem:** The `teaching_page.py` file uses `QRect` class without importing it, causing a NameError when trying to update the UI.

**Symptoms:**
```
NameError: name 'QRect' is not defined
```

**Root Cause:** Missing import statement for `QRect` from `PyQt5.QtCore`.

**Fix Applied:** Added `QRect` to the imports in `teaching_page.py`.

**Files Modified:**
- `src/ui/teaching_page.py` line 11

---

### 3. **RTSP URL Credential Malformation**
**Problem:** When entering RTSP credentials, double colons (`::`) were being used instead of single colon (`:`) in the username:password format.

**Symptoms:**
- First attempt: `rtsp://admin:Pass_123@192.168.1.64:554/stream` ✓ Works
- Later attempts: `rtsp://admin::Pass_123@192.168.1.64:554/stream` ✗ 401 Unauthorized

**Root Cause:** User error during manual URL entry - accidentally entering double colons or URL parsing not handling special characters in passwords correctly.

**Fix Applied:** Added RTSP URL validation method `_validate_rtsp_url()` that:
- Validates URL format
- Checks for proper credentials format (username:password)
- Detects if password contains colons which could cause parsing issues
- Provides clear error messages to the user

**Files Modified:**
- `src/ui/teaching_page.py` - Added `_validate_rtsp_url()` method before `_add_camera()`

---

### 4. **Excessive Connection Attempts / DDoS-like Behavior**
**Problem:** The camera continuously attempts to reconnect with exponential backoff, which combined with credential errors can create a high load on the camera server, potentially causing it to block connections for 30 minutes.

**Symptoms:**
- Reconnection attempts: 1s, 2s, 4s, 8s, 16s, 32s, ... (infinite loop)
- Camera gets blocked for extended period

**Root Cause:** 
1. No maximum attempt limit before giving up
2. Exponential backoff continues indefinitely
3. No clear diagnostic messages about why connection is failing

**Fixes Applied:**

#### In `src/camera/reconnect_policy.py`:
- Added `max_attempts` parameter (default: 10 attempts)
- Implemented exhaustion detection - stops retrying after max attempts
- Added detailed logging explaining common issues
- When exhausted, logs helpful debugging information:
  - Check for incorrect credentials (double colons)
  - Check IP address and port
  - Inform user camera may have blocked connections

#### In `src/camera/camera_worker.py`:
- Enhanced error logging with specific checks for users:
  - URL format validation
  - Credentials format check (double colon warning)
  - Camera accessibility check
  - Connection blocking detection

**Files Modified:**
- `src/camera/reconnect_policy.py` - Enhanced with attempt limiting
- `src/camera/camera_worker.py` - Improved error messages

---

## Prevention Strategies

### For Users:
1. **Enter RTSP URL carefully** - Double-check username:password format
   - ✓ Correct: `rtsp://admin:Pass_123@192.168.1.64:554/stream`
   - ✗ Wrong: `rtsp://admin::Pass_123@192.168.1.64:554/stream` (double colon)

2. **Check logs for detailed error messages** - The application now provides:
   - Specific validation errors for URL format
   - Clear diagnostics when connection fails
   - Exhaustion warning after 10 failed attempts

3. **If camera is blocked** - Wait for the 30-minute timeout to expire or:
   - Restart the camera hardware
   - Check camera's admin panel for blocked connection list

### For Developers:
1. Always test with invalid RTSP URLs to ensure proper error handling
2. Use character-safe logging on Windows (no Unicode special chars)
3. Implement attempt limiting for reconnection policies
4. Provide detailed error context in logs for debugging

---

## Testing Recommendations

1. **Test valid RTSP URL** - Should connect and stream
2. **Test invalid credentials** - Should fail gracefully after 10 attempts with clear error
3. **Test wrong IP/port** - Should provide helpful error message
4. **Test with special characters in password** - Should validate and warn if colons present
5. **Check Windows console** - No encoding errors should appear

---

## Files Changed Summary

| File | Changes | Reason |
|------|---------|--------|
| `src/app.py` | Replaced ✓ with [OK] | Windows console encoding compatibility |
| `src/ui/teaching_page.py` | Added QRect import + URL validation | Fix NameError and input validation |
| `src/camera/camera_worker.py` | Enhanced error messages | Better debugging information |
| `src/camera/reconnect_policy.py` | Added attempt limiting + exhaustion detection | Prevent DDoS-like behavior |

---

## How Camera Connection Now Works

1. User enters RTSP URL via dialog
2. URL is validated for correct format and credentials
3. If invalid, user gets specific error message to fix it
4. Camera attempts connection with exponential backoff
5. After 10 failed attempts, reconnection stops and logs detailed diagnostic
6. User must fix the issue (credentials, IP, etc.) and restart app
7. On successful connection, counters reset

This prevents continuous hammering of the camera server while providing clear feedback to the user.
