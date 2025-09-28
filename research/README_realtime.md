# Real-time Fish Recognition System

Sistem real-time untuk deteksi dan klasifikasi ikan berdasarkan kode penelitian `research_fishial.py`.

## Files yang Dibuat

### 1. `realtime_fish_recognition.py`
- Sistem real-time lengkap dengan semua fitur
- Deteksi ikan, segmentasi, dan klasifikasi
- Interface kamera dengan overlay informasi
- Fitur save hasil deteksi

### 2. `optimized_realtime_fish_recognition.py` 
- Versi yang dioptimasi untuk performa tinggi
- Threading untuk processing paralel
- Frame skipping untuk FPS yang lebih baik
- Opsi disable segmentasi untuk performa maksimal

### 3. `simple_fish_demo.py`
- Demo sederhana untuk testing
- Hanya deteksi dan klasifikasi ikan
- Minimal dependencies

## Instalasi

1. **Install dependencies:**
```bash
pip install -r requirements_realtime.txt
```

2. **Pastikan model sudah terdownload:**
   - Model akan otomatis didownload saat menjalankan `research_fishial.py` untuk pertama kali
   - Atau pastikan folder `models/` berisi:
     - `models/classification/`
     - `models/detection/`
     - `models/segmentation/`
     - `models/face_detector/`

## Cara Menggunakan

### Demo Sederhana (Recommended untuk testing):
```bash
python simple_fish_demo.py
```

### Real-time Lengkap:
```bash
python realtime_fish_recognition.py
```

### Versi Optimized (Performa Terbaik):
```bash
python optimized_realtime_fish_recognition.py
```

## Kontrol

- **Q**: Quit/keluar
- **S**: Save frame dan hasil deteksi
- **T**: Toggle segmentasi (pada versi optimized)

## Features

### Real-time Fish Recognition:
- ✅ Live camera feed
- ✅ Fish detection dengan YOLO
- ✅ Fish classification 
- ✅ Fish segmentation
- ✅ Face detection (opsional)
- ✅ FPS counter
- ✅ Bounding box dan labels
- ✅ Save hasil deteksi
- ✅ Performance monitoring

### Optimizations:
- ✅ Frame skipping untuk performa
- ✅ Threading untuk processing paralel
- ✅ Model warm-up
- ✅ Batch processing
- ✅ Memory management
- ✅ Configurable thresholds

## Struktur Output

Saat save (tekan 'S'):
- `fish_detection_YYYYMMDD_HHMMSS.jpg` - Frame dengan annotations
- `fish_results_YYYYMMDD_HHMMSS.json` - Data deteksi dalam JSON

Format JSON:
```json
{
  "timestamp": "20231216_143022",
  "fish_count": 2,
  "detections": [
    {
      "name": "Goldfish",
      "accuracy": 0.85,
      "bbox": [100, 150, 300, 350],
      "confidence": 0.92
    }
  ]
}
```

## Performance Tips

1. **Untuk FPS maksimal:**
   - Gunakan `optimized_realtime_fish_recognition.py`
   - Disable segmentasi dan face detection
   - Turunkan resolusi kamera jika perlu

2. **Untuk akurasi maksimal:**
   - Gunakan `realtime_fish_recognition.py` 
   - Enable semua fitur
   - Gunakan resolusi kamera tinggi

3. **Untuk testing/demo:**
   - Gunakan `simple_fish_demo.py`
   - Minimal resource usage

## Troubleshooting

1. **Camera tidak terbuka:**
   - Cek camera permissions
   - Ganti `camera_id` (0, 1, 2, dll)

2. **Model tidak ditemukan:**
   - Jalankan `research_fishial.py` dulu untuk download models
   - Pastikan folder `models/` ada dan berisi file model

3. **FPS rendah:**
   - Gunakan versi optimized
   - Disable segmentasi
   - Turunkan resolusi kamera

4. **Error torch/CUDA:**
   - Pastikan PyTorch terinstall dengan benar
   - Sistem menggunakan CPU secara default

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- NumPy
- Camera/webcam
- Model files (otomatis download)

## Modifikasi

File dapat dimodifikasi untuk:
- Ganti threshold deteksi
- Tambah jenis ikan baru
- Integrasi dengan database
- Export ke format lain
- Streaming ke network

Semua kode dibuat berdasarkan struktur dan workflow dari `research_fishial.py` dengan optimasi untuk real-time performance.