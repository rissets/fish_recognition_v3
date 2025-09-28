# Fish Recognition API

Django REST API aplikasi untuk mengenali ikan yang menggabungkan object detection, classification, dan segmentation dengan kemampuan real-time melalui WebSocket.

## Fitur Utama

- **Multi-Model Fish Recognition**: Menggabungkan model detection, classification, dan segmentation
- **Real-time Processing**: Dukungan WebSocket untuk live camera feed
- **High Accuracy**: Dioptimalkan untuk akurasi dengan adaptive processing
- **Batch Processing**: Pemrosesan efisien untuk multiple images
- **Performance Monitoring**: Metrics dan statistik yang komprehensif
- **Image Quality Validation**: Penilaian kualitas otomatis dan rekomendasi
- **Caching System**: Redis-based caching untuk performa yang lebih baik
- **Face Detection**: Kemampuan deteksi wajah ikan

## Quick Start

### 1. Setup Environment

```bash
# Clone atau copy project
cd fish_api/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Model Files

Pastikan model files tersedia di path yang benar:
```
models/
├── classification/
│   ├── model.ts
│   ├── database.pt
│   └── labels.json
├── detection/
│   └── model.ts
├── segmentation/
│   └── model.ts
└── face_detector/
    └── model.ts
```

### 3. Setup Redis (untuk caching dan WebSocket)

```bash
# Install Redis
brew install redis  # macOS
# atau
sudo apt-get install redis-server  # Ubuntu

# Start Redis
redis-server
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env sesuai kebutuhan
```

### 5. Database Migration

```bash
python manage.py migrate
```

### 6. Run Development Server

```bash
# HTTP + WebSocket server
python manage.py runserver
```

### 7. Access Testing App

Buka browser dan akses: `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /api/v1/health/
```

### Single Image Recognition
```
POST /api/v1/recognize/
```

### Batch Processing
```
POST /api/v1/recognize/batch/
```

### Performance Stats
```
GET /api/v1/stats/
```

### Model Configuration
```
GET/POST /api/v1/config/
```

## WebSocket API

### Connection
```javascript
// Full recognition pipeline (classification + segmentation)
const ws = new WebSocket('ws://localhost:8000/ws/recognition/');

// Lightweight detection stream (object presence only)
const detectionWs = new WebSocket('ws://localhost:8000/ws/recognition/detection/');
```

### Send Camera Frame
```javascript
ws.send(JSON.stringify({
    type: 'camera_frame',
    data: {
        frame_data: 'data:image/jpeg;base64,...',
        include_faces: true,
        include_segmentation: true
    }
}));
```

### Batch Capture for Aggregated Classification

Sejak optimisasi terbaru, aplikasi front-end mengumpulkan 10–60 frame terbaik saat ikan terdeteksi dan mengirimkannya sekaligus untuk meningkatkan akurasi akhir.

```javascript
ws.send(JSON.stringify({
    type: 'classification_batch',
    data: {
        frames: [ /* array base64 frame dari kamera */ ],
        include_faces: true,
        include_segmentation: true,
        include_visualization: true,
        quality_threshold: 0.3
    }
}));
```

Respons yang diterima bertipe `recognition_result` dengan payload hasil frame terbaik, plus `batch` dan `aggregate_summary` yang memuat ringkasan voting spesies, rasio mayoritas, dan statistik total deteksi.

## Testing Application

Testing app tersedia di root URL (`http://localhost:8000`) dengan fitur:

### 📷 Image Upload Mode
- Upload single image untuk analysis
- Drag & drop support
- Real-time preview

### 📹 Live Camera Mode  
- Akses camera device untuk real-time recognition
- WebSocket connection untuk live updates
- Adaptive processing (accuracy vs speed mode)
- Overlay annotations pada video
- Stream deteksi cepat + buffer 10–60 frame untuk voting klasifikasi akhir
- Panel guidance dengan status deteksi, confidence, dan indikator buffer
- Mode tambahan **Learning Without Forgetting (LWF)** untuk menambah embedding langsung dari UI

### 🧠 LWF Adaptation Mode
- Unggah hingga 60 gambar (drag & drop atau pilih manual) dengan dukungan semua format
- Pipeline ultra advanced: deteksi YOLO, multi-preprocessing (attention, edge enhancement, contrast, histogram, noise, multi-scale crop)
- Menonjolkan konsistensi embedding: menghitung majority vote & centroid shift terhadap data lama
- Setelah submit, model & label otomatis di-*reload* sehingga siap dipakai oleh seluruh endpoint

### 📚 Batch Processing Mode
- Upload multiple images (max 10)
- Batch analysis dengan progress tracking
- Detailed results per image

### ⚙️ Settings
- Include/exclude face detection
- Include/exclude segmentation  
- Visualization options
- Quality threshold adjustment
- Processing mode selection

### 📊 Live Statistics
- Real-time processing stats
- Success rate monitoring
- Performance metrics
- Connection status

## Logging

### Detection Stream Output

Konsumer WebSocket `ws/recognition/detection/` kini mencatat ringkasan setiap frame yang berhasil diproses, sehingga mudah melakukan troubleshooting langsung dari console Django:

```
2025-09-28 01:08:33,138 INFO recognition.consumers.detection_consumer Detection result - frame=1694823452345 has_fish=True count=1 max_conf=0.873 avg_conf=0.812 processing=0.042s quality_ok=True
```

Format log menampilkan `frame_id`, ada/tidaknya ikan (`has_fish`), jumlah deteksi (`count`), confidence maksimal/rata-rata, durasi pemrosesan, serta status validasi kualitas gambar (`quality_ok`).

## Architecture

```
fish_api/
├── fish_recognition_api/     # Django project settings
├── recognition/              # Main app
│   ├── ml_models/           # ML engine
│   ├── utils/               # Utilities (image, performance)
│   ├── consumers/           # WebSocket consumers
│   ├── views.py             # REST API views
│   ├── serializers.py       # API serializers
│   └── urls.py              # URL routing
├── static/                  # Static files (CSS, JS)
├── templates/               # HTML templates
├── media/                   # Media uploads
└── requirements.txt         # Dependencies
```

## Performance Optimization

### Caching
- Redis untuk model predictions (1 hour TTL)
- Memory cache untuk hasil recent
- SHA256 hashing untuk cache keys

### Image Processing
- Efficient preprocessing untuk setiap model type
- Batch processing support
- Memory management otomatis

### Model Optimization
- TorchScript pre-compiled models
- CPU-optimized inference
- Minimal memory footprint

## Production Deployment

### 1. Environment Variables
```bash
DEBUG=False
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=your-domain.com
REDIS_URL=redis://production-redis:6379/0
```

### 2. ASGI Server
```bash
# Install production server
pip install daphne gunicorn

# Run ASGI server (untuk WebSocket support)
daphne -b 0.0.0.0 -p 8000 fish_recognition_api.asgi:application
```

### 3. Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "fish_recognition_api.asgi:application"]
```

### 4. Nginx Configuration
```nginx
upstream fish_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://fish_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /path/to/staticfiles/;
    }
}
```

## Monitoring & Troubleshooting

### Logs
- Application logs: `fish_api.log`
- Django logs: console output
- WebSocket logs: included in application logs

### Common Issues

1. **Models Not Loading**
   - Check model file paths in settings
   - Verify PyTorch version compatibility
   - Ensure models directory accessible

2. **Poor Recognition Accuracy**
   - Check image quality (lighting, focus, resolution)  
   - Verify confidence thresholds
   - Ensure fish clearly visible

3. **Slow Processing**
   - Enable caching
   - Use batch processing
   - Monitor memory usage
   - Consider GPU acceleration

4. **WebSocket Issues**
   - Verify Redis running
   - Check CORS settings
   - Monitor connection timeouts

### Performance Monitoring
```bash
# Check API health
curl http://localhost:8000/api/v1/health/

# Get performance stats  
curl http://localhost:8000/api/v1/stats/
```

## API Documentation

Lengkap documentation tersedia di `API_DOCUMENTATION.md` yang mencakup:
- Detailed endpoint descriptions
- Request/response examples
- WebSocket message formats
- Error handling
- Performance optimization tips

## Support

Untuk troubleshooting atau pertanyaan teknis, periksa:
1. Application logs untuk error details
2. API health endpoint untuk status models
3. Redis connectivity untuk WebSocket issues
4. Image quality validation untuk recognition problems
