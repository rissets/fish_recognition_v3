# WebSocket Auto-Detection Update

## Summary
WebSocket consumer telah diperbarui dengan fitur **auto-detection** yang secara otomatis melakukan prediksi ketika ikan terdeteksi dalam **3 frame berturut-turut**, tanpa perlu action `analyze` atau `capture` manual.

---

## 🎯 Changes Made

### 1. **Consumer Updates** (`recognition/consumers/recognition_consumer.py`)

#### Added Auto-Detection Buffer
```python
# Buffer untuk tracking deteksi ikan
self.fish_detection_buffer = []
self.consecutive_fish_threshold = 3  # 3 frame berturut-turut
self.max_buffer_size = 5
```

#### New Setting
```python
self.client_settings = {
    ...
    'auto_detection_enabled': True  # Enable/disable feature
}
```

#### New Methods
- `_process_frame_with_auto_detection()`: Process frame dengan auto-detection logic
- `_quick_fish_detection()`: Lightweight detection (hanya YOLO, tanpa classification)
- `_count_consecutive_detections()`: Count consecutive True values di buffer

---

## 🔄 How It Works

### Flow Diagram
```
Frame Stream → Quick Detection → Buffer Update → Threshold Check → Full Recognition
     ↓              ↓                 ↓               ↓                  ↓
  Camera         YOLO only      [T,T,T,F,T]     3 consecutive?    Classification + LLM
```

### Step-by-Step
1. **Frame masuk** dari camera stream
2. **Quick detection**: Check dengan YOLO detection model saja (~50-100ms)
3. **Update buffer**: Simpan hasil (True/False) di buffer
4. **Count consecutive**: Hitung berapa frame berturut-turut ada ikan
5. **Threshold check**: Jika ≥ 3 frame berturut-turut → trigger full recognition
6. **Full recognition**: Detection + Classification + LLM (~5-6 seconds)
7. **Clear buffer**: Setelah recognition berhasil, buffer di-clear

---

## 📊 Performance

| Operation | Time | Resource |
|-----------|------|----------|
| Quick Detection | 50-100ms | Low CPU |
| Full Recognition | 5-6 seconds | High CPU + LLM |
| Frame Processing | ~200ms/frame | Minimal |

**Throttling**: Minimum 0.5 detik antara full recognition (mencegah overload)

---

## 📡 WebSocket Messages

### Client → Server (Stream Frames)
```json
{
  "type": "camera_frame",
  "data": {
    "frame_data": "base64_encoded_image",
    "frame_id": 123
  }
}
```

### Server → Client (Detection Status)
```json
{
  "type": "detection_status",
  "data": {
    "has_fish": true,
    "consecutive_count": 2,
    "threshold": 3,
    "buffer": [true, true, false]
  }
}
```

### Server → Client (Auto Recognition)
```json
{
  "type": "recognition_result",
  "data": {
    "source": "auto_detection",
    "trigger": "fish_detected_3_frames",
    "processing_time": 5.23,
    "results": {
      "fish_detections": [...],
      "classification": [
        {
          "name": "Ikan Bandeng",          // Indonesian name (PRIMARY)
          "scientific_name": "Chanos chanos",
          "accuracy": 0.95,
          "source": "llm",
          "species_id": -1
        }
      ]
    }
  }
}
```

---

## ⚙️ Configuration

### Enable/Disable Auto-Detection
```json
{
  "type": "settings_update",
  "data": {
    "auto_detection_enabled": true  // or false
  }
}
```

### Adjust Threshold (Server-Side)
```python
# In consumer __init__
self.consecutive_fish_threshold = 3  # Change to 2, 4, 5, etc.
```

---

## 🧪 Testing

### Test Script
```bash
cd fish_api
python test_websocket_auto_detection.py
```

### HTML Demo
1. Start Django server:
   ```bash
   cd fish_api
   python manage.py runserver
   ```

2. Open browser:
   ```
   http://localhost:8000/templates/websocket_demo.html
   ```

3. Click "Start Camera" and point at fish image
4. Watch auto-detection trigger after 3 frames!

---

## 🎨 UI Integration Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/recognition/');

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  
  if (msg.type === 'detection_status') {
    // Update detection indicator
    updateDetectionUI(msg.data.consecutive_count, msg.data.threshold);
  }
  
  if (msg.type === 'recognition_result') {
    // Display results (auto-triggered!)
    displayFishResults(msg.data.results);
  }
};
```

---

## ✅ Benefits

### User Experience
- ✅ **No manual actions**: Fully automatic
- ✅ **Real-time feedback**: Immediate detection status
- ✅ **Fast response**: Triggers as soon as fish stable in frame

### Performance
- ✅ **Resource efficient**: Quick detection is lightweight
- ✅ **Smart throttling**: Only full recognition when needed
- ✅ **No duplicates**: Buffer cleared after recognition

### Accuracy
- ✅ **3-frame confirmation**: Reduces false positives
- ✅ **LLM primary output**: Indonesian name first
- ✅ **Stable detection**: Only triggers when fish consistently present

---

## 🔧 Troubleshooting

### Recognition not triggering?
- Check `detection_status` messages
- Ensure `consecutive_count` reaches `threshold` (3)
- Verify `auto_detection_enabled` is `true`

### Too many recognitions?
- Increase `consecutive_fish_threshold` (e.g., 5)
- Increase `min_processing_interval` (e.g., 1.0 seconds)

### Detection too slow?
- Decrease `consecutive_fish_threshold` (e.g., 2)
- Change `processing_mode` to `"speed"`

---

## 📚 Files Modified

1. **recognition/consumers/recognition_consumer.py**
   - Added auto-detection logic
   - Added buffer management
   - Added quick detection method

2. **WEBSOCKET_AUTO_DETECTION.md**
   - Comprehensive documentation

3. **test_websocket_auto_detection.py**
   - Test script for WebSocket

4. **templates/websocket_demo.html**
   - Interactive HTML demo

---

## 🚀 Next Steps

Sistem sudah siap digunakan! Untuk testing:

1. **Command-line test**:
   ```bash
   cd fish_api
   python test_websocket_auto_detection.py
   ```

2. **Browser demo**:
   - Start server: `python manage.py runserver`
   - Open: `http://localhost:8000/templates/websocket_demo.html`
   - Click "Start Camera"
   - Point camera at fish image
   - Watch auto-detection work!

---

## 📖 Documentation

- Detailed guide: `WEBSOCKET_AUTO_DETECTION.md`
- LLM integration: `LLM_PRIMARY_OUTPUT_UPDATE.md`
- API docs: `API_DOCUMENTATION.md`
