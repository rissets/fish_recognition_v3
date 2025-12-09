# Fish Identification & Dataset Management System

Sistem terintegrasi untuk identifikasi ikan yang **otomatis menyimpan** hasil prediksi AI ke database, memungkinkan user untuk mengoreksi jika salah, dan export dataset untuk training model baru.

## 🔄 Alur Kerja Sistem

```
User Upload Gambar
    ↓
AI Model Process (Detection + Classification + LLM)
    ↓
OTOMATIS SIMPAN ke Database + MinIO
    ↓
Response ke User (dengan identification_id)
    ↓
[OPTIONAL] User Review & Correct jika salah
    ↓
Data Ready untuk Export sebagai Training Dataset
```

## ✨ Fitur Utama

### 1. **Auto-Save After Recognition** 🆕
- Setiap kali AI berhasil identifikasi → **otomatis disimpan** ke database
- Gambar di-upload ke MinIO storage
- Thumbnail otomatis dibuat
- Metadata lengkap (confidence, detection box, KB candidates)

### 2. **User Correction System**
- User dapat mengoreksi nama ikan jika AI salah
- History tracking semua perubahan
- Status management (pending/verified/corrected/rejected)

### 3. **Dataset Export for Training** 🆕
- Export ke JSON/CSV/YOLO format
- Filter berdasarkan status, kelompok, confidence
- Pilih pakai nama original AI atau corrected names
- Statistik dataset lengkap

## 📡 API Endpoints

### 1. Recognition (Auto-Save) ⭐

**POST** `/api/recognition/recognize/`

Endpoint ini sekarang **otomatis menyimpan** hasil ke database!

Request:
```bash
curl -X POST http://localhost:8000/api/recognition/recognize/ \
  -F "image=@fish.jpg" \
  -F "user_identifier=user123" \
  -F "user_location=Jakarta"
```

Response:
```json
{
  "success": true,
  "identification_id": "uuid-generated",  // 🆕 ID untuk tracking
  "image_shape": [1024, 768, 3],
  "fish_detections": [
    {
      "id": 0,
      "bbox": [100, 150, 300, 400],
      "confidence": 0.95,
      "classification": [
        {
          "name": "Scomberomorus maculatus",
          "label": "Tenggiri atlantik",
          "accuracy": 0.95
        }
      ]
    }
  ],
  "processing_time": {...}
}
```

**Note:** Data otomatis tersimpan di database dengan status `pending`.

### 2. User Correction Endpoint ⭐

**POST** `/api/recognition/identifications/{id}/correct/`

User hanya menggunakan endpoint ini untuk **mengoreksi nama** jika AI salah.

Request:
```json
{
  "scientific_name": "Rastrelliger kanagurta",
  "indonesian_name": "Kembung lelaki",
  "english_name": "indian mackerel",
  "kelompok": "Kembung",
  "notes": "AI prediksi salah, ini kembung lelaki bukan kembung perempuan"
}
```

Response:
```json
{
  "id": "uuid",
  "original_indonesian_name": "Kembung perempuan",
  "current_indonesian_name": "Kembung lelaki",
  "is_corrected": true,
  "status": "corrected",
  "corrected_at": "2025-12-09T11:00:00Z"
}
```

### 3. Export Dataset for Training 🆕

**GET** `/api/recognition/dataset/export/`

Export data untuk training model baru.

Query Parameters:
- `format`: `json` (default), `csv`, `yolo`
- `status`: Filter by status (default: verified,corrected)
- `kelompok`: Filter by kelompok
- `min_confidence`: Minimum confidence (e.g., 0.7)
- `use_corrected`: `true` (default) - gunakan nama yang sudah dikoreksi

**Example 1: Export JSON (High Quality)**
```bash
curl "http://localhost:8000/api/recognition/dataset/export/?format=json&min_confidence=0.8&use_corrected=true"
```

Response:
```json
{
  "total_records": 150,
  "export_format": "json",
  "use_corrected_names": true,
  "filters": {
    "status": "verified,corrected",
    "min_confidence": "0.8"
  },
  "dataset": [
    {
      "id": "uuid",
      "image_url": "http://minio:9000/fish-media/fish_images/2025/12/09/uuid.jpg",
      "scientific_name": "Scomberomorus maculatus",
      "indonesian_name": "Tenggiri atlantik",
      "english_name": "atlantic spanish mackerel",
      "kelompok": "Tenggiri",
      "confidence_score": 0.95,
      "detection_box": [100, 150, 300, 400],
      "status": "verified",
      "is_corrected": false,
      "was_ai_correct": true,
      "created_at": "2025-12-09T10:00:00Z"
    }
  ]
}
```

**Example 2: Export CSV**
```bash
curl "http://localhost:8000/api/recognition/dataset/export/?format=csv&kelompok=Tenggiri" > fish_dataset.csv
```

**Example 3: Filter Specific Kelompok**
```bash
# Export only Udang data for training
curl "http://localhost:8000/api/recognition/dataset/export/?kelompok=Udang&min_confidence=0.7"

# Export only Kepiting and Rajungan
curl "http://localhost:8000/api/recognition/dataset/export/?kelompok=Kepiting"
```

### 4. Dataset Statistics 🆕

**GET** `/api/recognition/dataset/statistics/`

Statistik lengkap untuk dataset training.

Response:
```json
{
  "total_identifications": 500,
  "training_ready": 450,
  "by_status": [
    {"status": "verified", "count": 300},
    {"status": "corrected", "count": 150},
    {"status": "pending", "count": 50}
  ],
  "by_kelompok": [
    {"current_kelompok": "Tenggiri", "count": 100, "avg_confidence": 0.92},
    {"current_kelompok": "Kembung", "count": 80, "avg_confidence": 0.88},
    {"current_kelompok": "Udang", "count": 70, "avg_confidence": 0.85}
  ],
  "correction_statistics": {
    "total_corrected": 150,
    "correction_rate": 30.0
  },
  "ai_accuracy": 70.0,
  "top_20_species": [
    {
      "current_indonesian_name": "Tenggiri atlantik",
      "current_scientific_name": "Scomberomorus maculatus",
      "count": 50
    }
  ],
  "confidence_distribution": {
    "very_high (0.9-1.0)": 250,
    "high (0.7-0.9)": 150,
    "medium (0.5-0.7)": 80,
    "low (<0.5)": 20
  }
}
```

### 5. Other Endpoints

**List All Identifications**
```bash
GET /api/recognition/identifications/?status=verified&ordering=-confidence_score
```

**Get Detail**
```bash
GET /api/recognition/identifications/{id}/
```

**Verify (Mark as Correct)**
```bash
POST /api/recognition/identifications/{id}/verify/
```

**Reject**
```bash
POST /api/recognition/identifications/{id}/reject/
```

**View History**
```bash
GET /api/recognition/identifications/{id}/history/
```

**Species Statistics**
```bash
GET /api/recognition/species/statistics/?ordering=-total_identifications
```

## 🎯 Use Cases

### Use Case 1: User Upload & Auto-Save
```python
import requests

# User upload gambar
files = {'image': open('fish.jpg', 'rb')}
data = {
    'user_identifier': 'user123',
    'user_location': 'Jakarta'
}

response = requests.post(
    'http://localhost:8000/api/recognition/recognize/',
    files=files,
    data=data
)

result = response.json()
print(f"Identification ID: {result['identification_id']}")
print(f"Predicted: {result['fish_detections'][0]['classification'][0]['label']}")

# ✅ Data sudah otomatis tersimpan di database!
```

### Use Case 2: User Koreksi Nama yang Salah
```python
import requests

identification_id = "uuid-from-recognition"

# User koreksi nama
correction_data = {
    "scientific_name": "Rastrelliger kanagurta",
    "indonesian_name": "Kembung lelaki",
    "notes": "AI salah, ini kembung lelaki"
}

response = requests.post(
    f'http://localhost:8000/api/recognition/identifications/{identification_id}/correct/',
    json=correction_data
)

# ✅ Nama sudah diupdate, history tercatat
```

### Use Case 3: Export Dataset untuk Training
```python
import requests
import json

# Export high-quality dataset
params = {
    'format': 'json',
    'min_confidence': 0.8,
    'use_corrected': 'true',  # Pakai nama yang sudah dikoreksi
    'status': 'verified,corrected'
}

response = requests.get(
    'http://localhost:8000/api/recognition/dataset/export/',
    params=params
)

dataset = response.json()
print(f"Total training data: {dataset['total_records']}")

# Save to file
with open('training_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

# ✅ Dataset ready untuk training!
```

### Use Case 4: Monitor Dataset Quality
```python
import requests

# Get statistics
response = requests.get('http://localhost:8000/api/recognition/dataset/statistics/')
stats = response.json()

print(f"Total data: {stats['total_identifications']}")
print(f"Training ready: {stats['training_ready']}")
print(f"AI Accuracy: {stats['ai_accuracy']}%")
print(f"Correction rate: {stats['correction_statistics']['correction_rate']}%")

# ✅ Monitor kualitas dataset
```

## 🗄️ Database Schema

### FishIdentification Model

```python
{
    "id": UUID,  # Primary key
    
    # Images
    "image": ImageField,  # Original image
    "image_url": String,  # MinIO URL
    "thumbnail": ImageField,  # Thumbnail
    
    # Original AI Prediction (IMMUTABLE)
    "original_scientific_name": String,
    "original_indonesian_name": String,
    "original_english_name": String,
    "original_kelompok": String,
    
    # Current (Can be corrected by user)
    "current_scientific_name": String,
    "current_indonesian_name": String,
    "current_english_name": String,
    "current_kelompok": String,
    
    # AI Metadata
    "confidence_score": Float,
    "ai_model_version": String,
    "detection_box": JSON [x, y, w, h],
    "kb_candidates": JSON,
    
    # Status & Tracking
    "status": Enum (pending/verified/corrected/rejected),
    "is_corrected": Boolean,
    "correction_notes": Text,
    
    # User Info
    "user_identifier": String,
    "user_location": String,
    
    # Timestamps
    "created_at": DateTime,
    "updated_at": DateTime,
    "corrected_at": DateTime
}
```

## 📊 Dataset Export Formats

### 1. JSON Format (Recommended)
```json
{
  "id": "uuid",
  "image_url": "http://...",
  "scientific_name": "...",
  "indonesian_name": "...",
  "kelompok": "...",
  "confidence_score": 0.95,
  "detection_box": [x, y, w, h]
}
```

### 2. CSV Format
```csv
id,image_url,scientific_name,indonesian_name,kelompok,confidence_score
uuid,http://...,Scomberomorus maculatus,Tenggiri atlantik,Tenggiri,0.95
```

### 3. YOLO Format (for Object Detection)
```json
{
  "classes": {
    "Tenggiri atlantik": 0,
    "Kembung lelaki": 1
  },
  "images": [
    {
      "id": "uuid",
      "url": "http://...",
      "label": "0 0.5 0.5 0.3 0.4"
    }
  ]
}
```

## 🔧 Setup & Migration

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Migrations
```bash
cd fish_api
python manage.py makemigrations
python manage.py migrate
```

### 3. Start MinIO
```bash
docker-compose up -d minio
```

### 4. Configure Environment
```bash
# .env
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=fish-media
```

### 5. Run Server
```bash
python manage.py runserver
```

## 📈 Dataset Quality Metrics

### Metrics yang Ditrack:
1. **AI Accuracy**: Berapa persen AI benar
2. **Correction Rate**: Berapa sering user koreksi
3. **Confidence Distribution**: Distribusi confidence score
4. **Species Distribution**: Species mana yang paling sering muncul
5. **Status Distribution**: pending/verified/corrected/rejected

### Quality Thresholds untuk Training:
- ✅ **High Quality**: `confidence >= 0.9` + `status=verified`
- ✅ **Good Quality**: `confidence >= 0.7` + `status=verified/corrected`
- ⚠️ **Medium Quality**: `confidence >= 0.5` + `status=corrected`
- ❌ **Low Quality**: `confidence < 0.5` atau `status=pending/rejected`

## 🚀 Production Checklist

- [ ] PostgreSQL configured (not SQLite)
- [ ] MinIO with proper backup strategy
- [ ] Regular database backups
- [ ] Monitoring untuk storage usage
- [ ] CDN untuk image serving
- [ ] Rate limiting untuk API
- [ ] Authentication untuk sensitive endpoints
- [ ] Periodic cleanup job untuk old data
- [ ] Dataset versioning system
- [ ] Training pipeline automation

## 📝 Notes

- **Data Immutability**: Original AI prediction TIDAK PERNAH diubah, hanya `current_*` fields yang bisa diupdate
- **History Tracking**: Semua perubahan tercatat di `FishIdentificationHistory`
- **Dataset Versioning**: Gunakan `created_at` dan `updated_at` untuk versioning
- **Quality Control**: Filter dengan `min_confidence` dan `status` untuk quality control
- **Privacy**: Jangan simpan PII (Personally Identifiable Information) di `user_identifier`

## 🔗 Related Files

- `recognition/models.py` - Database models
- `recognition/views.py` - API views dengan auto-save logic
- `recognition/services/minio_service.py` - MinIO storage service
- `recognition/urls.py` - URL routing
