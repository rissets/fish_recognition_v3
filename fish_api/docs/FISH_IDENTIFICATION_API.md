# Fish Identification Management API

Sistem untuk menyimpan hasil identifikasi ikan dengan gambar di MinIO dan memungkinkan user untuk mengoreksi nama ikan jika AI salah.

## 📋 Fitur Utama

1. **MinIO Storage** - Penyimpanan gambar di MinIO/S3
2. **Database Models** - Menyimpan hasil identifikasi dengan metadata lengkap
3. **Correction System** - User dapat mengoreksi nama ikan jika salah
4. **History Tracking** - Tracking semua perubahan yang dilakukan
5. **Species Statistics** - Statistik akurasi per species

## 🗄️ Database Models

### FishIdentification
Menyimpan hasil identifikasi ikan dengan detail lengkap:
- **Image Storage**: URL gambar di MinIO, thumbnail
- **Original AI Result**: Nama scientific/Indonesian yang diprediksi AI
- **Current Result**: Nama terkini (bisa sudah dikoreksi)
- **Metadata**: Confidence score, detection box, KB candidates
- **Status**: pending/verified/corrected/rejected
- **Tracking**: User identifier, timestamps, correction notes

### FishIdentificationHistory
Tracking semua perubahan:
- Field yang diubah
- Value lama dan baru
- User yang mengubah
- Timestamp dan alasan perubahan

### FishSpeciesStatistics
Statistik agregat per species:
- Total identifikasi
- Jumlah correct/corrected
- Average confidence
- Accuracy rate

## 🔧 Setup MinIO

### 1. Jalankan MinIO dengan Docker Compose

```bash
# MinIO sudah termasuk di docker-compose.yml
docker-compose up -d minio

# Akses MinIO Console
# http://localhost:9001
# Username: minioadmin
# Password: minioadmin123
```

### 2. Konfigurasi Environment Variables

Tambahkan di `.env`:
```bash
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=fish-media
MINIO_USE_SSL=False

# PostgreSQL (recommended for production)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=fish_recognition
POSTGRES_USER=fish_user
POSTGRES_PASSWORD=fish_password
```

### 3. Run Migrations

```bash
cd fish_api
python manage.py makemigrations
python manage.py migrate
```

### 4. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

## 📡 API Endpoints

### 1. List Identifications
**GET** `/api/recognition/identifications/`

Query Parameters:
- `status`: Filter by status (pending/verified/corrected/rejected)
- `is_corrected`: Filter by correction status (true/false)
- `kelompok`: Filter by kelompok
- `date_from`, `date_to`: Date range filter
- `search`: Search by name
- `ordering`: Sort by created_at, confidence_score, etc.
- `page`: Page number (pagination)

Response:
```json
{
  "count": 100,
  "next": "http://api/identifications/?page=2",
  "previous": null,
  "results": [
    {
      "id": "uuid",
      "image_url": "http://minio:9000/fish-media/...",
      "current_indonesian_name": "Tenggiri atlantik",
      "current_scientific_name": "Scomberomorus maculatus",
      "original_indonesian_name": "Tenggiri atlantik",
      "confidence_score": 0.95,
      "status": "verified",
      "is_corrected": false,
      "created_at": "2025-12-09T10:00:00Z",
      "was_ai_correct": true
    }
  ]
}
```

### 2. Get Identification Detail
**GET** `/api/recognition/identifications/{id}/`

Response:
```json
{
  "id": "uuid",
  "image_url": "http://minio:9000/fish-media/...",
  "thumbnail": "http://minio:9000/fish-media/...",
  "original_scientific_name": "Scomberomorus maculatus",
  "original_indonesian_name": "Tenggiri atlantik",
  "original_english_name": "atlantic spanish mackerel",
  "original_kelompok": "Tenggiri",
  "current_scientific_name": "Scomberomorus maculatus",
  "current_indonesian_name": "Tenggiri atlantik",
  "current_english_name": "atlantic spanish mackerel",
  "current_kelompok": "Tenggiri",
  "confidence_score": 0.95,
  "ai_model_version": "v1.0",
  "detection_box": [100, 150, 300, 400],
  "detection_score": 0.98,
  "kb_candidates": [...],
  "status": "verified",
  "is_corrected": false,
  "correction_notes": null,
  "user_identifier": "user123",
  "user_location": "Jakarta",
  "created_at": "2025-12-09T10:00:00Z",
  "updated_at": "2025-12-09T10:00:00Z",
  "corrected_at": null,
  "was_ai_correct": true
}
```

### 3. Correct Fish Identification ⭐
**POST** `/api/recognition/identifications/{id}/correct/`

Request Body:
```json
{
  "scientific_name": "Rastrelliger kanagurta",
  "indonesian_name": "Kembung lelaki",
  "english_name": "indian mackerel",
  "kelompok": "Kembung",
  "notes": "AI salah identifikasi, ini adalah kembung lelaki bukan kembung perempuan"
}
```

Response:
```json
{
  "id": "uuid",
  "current_scientific_name": "Rastrelliger kanagurta",
  "current_indonesian_name": "Kembung lelaki",
  "is_corrected": true,
  "status": "corrected",
  "corrected_at": "2025-12-09T11:00:00Z",
  ...
}
```

### 4. Verify Identification
**POST** `/api/recognition/identifications/{id}/verify/`

Mark identification as verified (AI was correct).

Response:
```json
{
  "id": "uuid",
  "status": "verified",
  ...
}
```

### 5. Reject Identification
**POST** `/api/recognition/identifications/{id}/reject/`

Request Body:
```json
{
  "notes": "Gambar tidak jelas, tidak bisa diidentifikasi"
}
```

### 6. Get Identification History
**GET** `/api/recognition/identifications/{id}/history/`

Response:
```json
[
  {
    "id": 1,
    "identification": "uuid",
    "field_name": "indonesian_name",
    "old_value": "Kembung perempuan",
    "new_value": "Kembung lelaki",
    "changed_by": "user123",
    "changed_at": "2025-12-09T11:00:00Z",
    "change_reason": "AI salah identifikasi"
  }
]
```

### 7. Species Statistics
**GET** `/api/recognition/species/statistics/`

Query Parameters:
- `kelompok`: Filter by kelompok
- `min_identifications`: Minimum number of identifications
- `search`: Search by name
- `ordering`: Sort by total_identifications, accuracy_rate, etc.

Response:
```json
{
  "count": 50,
  "results": [
    {
      "id": 1,
      "scientific_name": "Scomberomorus maculatus",
      "indonesian_name": "Tenggiri atlantik",
      "english_name": "atlantic spanish mackerel",
      "kelompok": "Tenggiri",
      "total_identifications": 150,
      "correct_identifications": 145,
      "corrected_identifications": 5,
      "average_confidence": 0.93,
      "accuracy_rate": 96.67,
      "first_seen": "2025-01-01T00:00:00Z",
      "last_seen": "2025-12-09T10:00:00Z"
    }
  ]
}
```

### 8. MinIO Health Check
**GET** `/api/recognition/minio/health/`

Response:
```json
{
  "status": "healthy",
  "endpoint": "localhost:9000",
  "bucket": "fish-media",
  "total_buckets": 1
}
```

## 🔄 Integration dengan Recognition API

Setelah melakukan recognition, simpan hasil ke database:

```python
from recognition.models import FishIdentification
from recognition.services.minio_service import get_minio_service

# Process image
results = engine.process_image(image_data)

# Upload to MinIO
minio = get_minio_service()
image_url = minio.upload_image(image_file, f"fish_images/2025/12/09/{uuid}.jpg")
thumbnail_bytes = minio.create_thumbnail(image_file)
thumbnail_url = minio.upload_image(thumbnail_bytes, f"fish_thumbnails/2025/12/09/{uuid}.jpg")

# Save to database
identification = FishIdentification.objects.create(
    image=image_file,
    image_url=image_url,
    thumbnail=thumbnail_url,
    original_scientific_name=llm_result['scientific_name'],
    original_indonesian_name=llm_result['indonesian_name'],
    original_english_name=kb_match['english_name'],
    original_kelompok=kb_match['kelompok'],
    current_scientific_name=llm_result['scientific_name'],
    current_indonesian_name=llm_result['indonesian_name'],
    current_english_name=kb_match['english_name'],
    current_kelompok=kb_match['kelompok'],
    confidence_score=results['fish_detections'][0]['classification'][0]['accuracy'],
    detection_box=results['fish_detections'][0]['bbox'],
    kb_candidates=kb_context['similar_species'],
    user_identifier=request.user.id,
    status='pending'
)
```

## 📊 Use Cases

### 1. User Melihat History Identifikasi
```bash
GET /api/recognition/identifications/?page=1&ordering=-created_at
```

### 2. User Mengoreksi Nama Ikan yang Salah
```bash
POST /api/recognition/identifications/{id}/correct/
{
  "scientific_name": "Correct Name",
  "indonesian_name": "Nama Yang Benar",
  "notes": "AI salah, ini sebenarnya..."
}
```

### 3. Admin Melihat Statistik Akurasi
```bash
GET /api/recognition/species/statistics/?ordering=-total_identifications
```

### 4. Filter Identifikasi yang Perlu Review
```bash
GET /api/recognition/identifications/?status=pending&ordering=confidence_score
```

### 5. Tracking Perubahan
```bash
GET /api/recognition/identifications/{id}/history/
```

## 🔐 Admin Interface

Access Django Admin:
```
http://localhost:8000/admin/
```

Models tersedia di admin:
- Fish Identifications
- Identification History
- Species Statistics

## 🧪 Testing

Test MinIO connection:
```bash
curl http://localhost:8000/api/recognition/minio/health/
```

Test identification APIs:
```bash
# List identifications
curl http://localhost:8000/api/recognition/identifications/

# Get detail
curl http://localhost:8000/api/recognition/identifications/{uuid}/

# Correct identification
curl -X POST http://localhost:8000/api/recognition/identifications/{uuid}/correct/ \
  -H "Content-Type: application/json" \
  -d '{
    "scientific_name": "New Name",
    "indonesian_name": "Nama Baru",
    "notes": "Correction reason"
  }'
```

## 📈 Performance Monitoring

Statistics yang bisa dimonitor:
1. **Total Identifications**: Berapa banyak ikan yang sudah diidentifikasi
2. **Accuracy Rate**: Persentase identifikasi yang benar
3. **Correction Rate**: Berapa sering user mengoreksi
4. **Species Distribution**: Species mana yang paling sering muncul
5. **Confidence Trends**: Trend confidence score per species

## 🚀 Production Tips

1. **Use PostgreSQL**: Lebih reliable untuk production
2. **MinIO Replication**: Setup MinIO dengan replication untuk high availability
3. **Backup Strategy**: Regular backup untuk database dan MinIO
4. **Monitoring**: Setup monitoring untuk MinIO dan database
5. **CDN**: Gunakan CDN untuk serving images dari MinIO
6. **Compression**: Enable image compression untuk save storage
7. **Cleanup Job**: Periodic cleanup untuk old/rejected identifications

## 🔗 Related Documentation

- [KNOWLEDGE_BASE_INTEGRATION.md](./KNOWLEDGE_BASE_INTEGRATION.md) - Knowledge base system
- [TEST_INTEGRATION_README.md](./TEST_INTEGRATION_README.md) - Testing guide
- [Docker Setup](../DOCKER_SETUP.md) - Docker configuration
