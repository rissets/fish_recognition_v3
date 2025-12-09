# WebSocket Auto-Detection: 2-Stage Process

## Overview
WebSocket auto-detection menggunakan **2-stage process** untuk efisiensi maksimal:

1. **Stage 1: Quick Detection** (YOLO only, ~50-100ms)
2. **Stage 2: Full Recognition** (Detection + Classification + Segmentation + LLM, ~5-6s)

---

## 🎯 Stage 1: Quick Detection (Lightweight)

### Purpose
- Check apakah ada ikan dalam frame
- Sangat cepat untuk real-time streaming
- Tidak melakukan classification atau segmentation

### What it does
```python
# HANYA detection dengan YOLO
detections = engine.detect_fish(image_bgr)
has_fish = len(detections) > 0
```

### Performance
- **Speed**: 50-100ms per frame
- **CPU**: Low (hanya YOLO detection)
- **Models used**: Detection model only
- **Output**: Boolean (True/False)

### What is NOT done
- ❌ Classification (tidak identify species)
- ❌ Segmentation (tidak extract mask)
- ❌ LLM verification (tidak panggil Ollama)
- ❌ Face detection

---

## 🔬 Stage 2: Full Recognition (Complete)

### Purpose
- Identify species dengan classification model
- Extract mask dengan segmentation model
- Verify dengan LLM (Ollama gamma3)
- Triggered ONLY after 3 consecutive detections

### What it does
```python
# FULL PIPELINE
results = engine.process_image(
    image_data=image_bytes,
    include_faces=True,
    include_segmentation=True
)
```

### Pipeline Steps
1. **Detection** (YOLO) → Find fish bounding boxes
2. **Classification** (BEiT-v2) → Identify species (639 classes)
3. **Segmentation** (SAM) → Extract fish mask
4. **LLM** (Ollama gamma3:27b) → Verify identification (Indonesian name)

### Performance
- **Speed**: 5-6 seconds per image
- **CPU**: High (all models + LLM inference)
- **Models used**: Detection + Classification + Segmentation + LLM
- **Output**: Complete fish recognition results

---

## 📊 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    WEBSOCKET FRAME STREAM                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: QUICK DETECTION (50-100ms)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  YOLO Detection Only                                     │   │
│  │  - Check if fish present                                 │   │
│  │  - NO classification                                     │   │
│  │  - NO segmentation                                       │   │
│  │  - NO LLM                                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    UPDATE DETECTION BUFFER                       │
│  [True, True, False, True, True, True]                          │
│   └────────────────────────────┬─────┘                          │
│              Count consecutive: 3/3 ✓                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Threshold reached (3 frames)?
                              ↓
                            YES
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             STAGE 2: FULL RECOGNITION (5-6 seconds)              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Detection (YOLO)        → Bounding boxes            │   │
│  │  2. Classification (BEiT)   → Species identification    │   │
│  │  3. Segmentation (SAM)      → Fish mask                 │   │
│  │  4. LLM (Ollama gamma3)     → Indonesian name           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      SEND RESULTS TO CLIENT                      │
│  {                                                               │
│    "type": "recognition_result",                                │
│    "source": "auto_detection",                                  │
│    "trigger": "fish_detected_3_frames",                         │
│    "results": {                                                 │
│      "classification": [                                        │
│        {                                                        │
│          "name": "Ikan Bandeng",  ← LLM (Indonesian name)      │
│          "scientific_name": "Chanos chanos",                   │
│          "source": "llm"                                        │
│        }                                                        │
│      ]                                                          │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Frame-by-Frame Example

### Scenario: Fish appears and stays in frame

```
Frame 1:
  Stage 1 → Quick Detection → Fish found ✓
  Buffer: [True]
  Consecutive: 1/3
  Action: Continue streaming (no full recognition yet)

Frame 2:
  Stage 1 → Quick Detection → Fish found ✓
  Buffer: [True, True]
  Consecutive: 2/3
  Action: Continue streaming (no full recognition yet)

Frame 3:
  Stage 1 → Quick Detection → Fish found ✓
  Buffer: [True, True, True]
  Consecutive: 3/3 ← THRESHOLD MET!
  Action: TRIGGER STAGE 2 (Full Recognition)
  
  Stage 2 → Full Recognition (5-6 seconds):
    ✓ Detection
    ✓ Classification
    ✓ Segmentation
    ✓ LLM verification
  
  Result: {
    "name": "Ikan Bandeng",
    "scientific_name": "Chanos chanos",
    "source": "llm"
  }
  
  Buffer cleared: []

Frame 4+:
  Continue streaming...
  If fish detected again in 3 frames → Trigger again
```

---

## 💡 Why 2-Stage Process?

### Without 2-Stage (Naive Approach)
```
Frame 1: Full Recognition (5s) → CPU 100%
Frame 2: Full Recognition (5s) → CPU 100%
Frame 3: Full Recognition (5s) → CPU 100%
Frame 4: Full Recognition (5s) → CPU 100%
...

Result: System overload, can't process real-time stream
```

### With 2-Stage (Smart Approach)
```
Frame 1: Quick Detection (0.1s) → CPU 20%
Frame 2: Quick Detection (0.1s) → CPU 20%
Frame 3: Quick Detection (0.1s) → CPU 20% → TRIGGER
Frame 3: Full Recognition (5s) → CPU 100% (only once)
Frame 4: Quick Detection (0.1s) → CPU 20%
...

Result: Efficient real-time processing with accurate recognition
```

### Benefits
- ✅ **Efficient**: 50x faster for quick checks
- ✅ **Real-time**: Can process streaming video
- ✅ **Accurate**: Full recognition only when needed
- ✅ **Smart**: 3-frame confirmation reduces false positives

---

## 🎛️ Configuration

### Consecutive Frame Threshold
```python
# In consumer __init__
self.consecutive_fish_threshold = 3  # Default: 3 frames

# Change to:
self.consecutive_fish_threshold = 2  # More sensitive (trigger faster)
self.consecutive_fish_threshold = 5  # Less sensitive (require more confirmation)
```

### Processing Mode
```json
{
  "type": "settings_update",
  "data": {
    "processing_mode": "speed"  // or "accuracy"
  }
}
```

**Speed mode**:
- Min interval: 0.1s
- Quality threshold: 0.2

**Accuracy mode**:
- Min interval: 0.5s
- Quality threshold: 0.3

---

## 📝 Log Output

### Stage 1 (Quick Detection)
```
[DEBUG] 🔍 Quick detection (YOLO only): ✓ Fish found (2 objects)
[INFO] Detection status: 2/3 consecutive frames
```

### Stage 2 (Full Recognition)
```
[INFO] 🎯 Auto-detection TRIGGERED! Fish detected in 3 consecutive frames
[INFO] ⚡ Starting FULL RECOGNITION: Detection + Classification + Segmentation + LLM
[INFO] 🔬 FULL RECOGNITION started: Detection → Classification → Segmentation → LLM
[INFO] 🐟 Running detection model...
[INFO] 🔬 Running classification model...
[INFO] ✂️  Running segmentation model...
[INFO] 🤖 Running LLM verification...
[INFO] ✅ FULL RECOGNITION completed successfully
[INFO] LLM identified fish 0: Ikan Bandeng (Chanos chanos)
```

---

## 🧪 Testing

### Test dengan static image
```bash
cd fish_api
python test_websocket_auto_detection.py
```

### Test dengan webcam
```bash
cd fish_api
python test_websocket_with_webcam.py
```

### Expected behavior:
1. Stream starts → Quick detection every frame
2. Fish appears → Buffer fills: 0/3, 1/3, 2/3
3. Threshold met → Full recognition triggered
4. Results shown → Indonesian name from LLM
5. Buffer cleared → Ready for next fish

---

## 🔧 Troubleshooting

### Quick detection too sensitive?
- Increase `consecutive_fish_threshold` to 5
- Quick detection will still be fast, but require more confirmation

### Full recognition too slow?
- Disable segmentation: `include_segmentation: false`
- Use speed mode: `processing_mode: "speed"`

### Not detecting fish?
- Check YOLO detection threshold in model config
- Test with clear fish image first
- Check logs for detection count

---

## 📚 Related Files

1. **recognition/consumers/recognition_consumer.py**
   - `_quick_fish_detection()` → Stage 1
   - `process_frame()` → Stage 2
   
2. **recognition/ml_models/fish_engine.py**
   - `detect_fish()` → YOLO detection only
   - `process_image()` → Full pipeline

3. **Test scripts**
   - `test_websocket_auto_detection.py` → Static image test
   - `test_websocket_with_webcam.py` → Live webcam test
