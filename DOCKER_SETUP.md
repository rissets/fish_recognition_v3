# Docker Setup Guide

## Layanan yang Tersedia

Docker Compose ini menyediakan 3 layanan utama:
1. **PostgreSQL** - Database untuk menyimpan data aplikasi
2. **Redis** - Cache dan message broker untuk Django Channels
3. **MinIO** - Object storage untuk media files (gambar ikan, dll)

## Quick Start

### 1. Setup Environment Variables

Copy file `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```

Edit `.env` sesuai kebutuhan Anda.

### 2. Start Services

Jalankan semua layanan:
```bash
docker-compose up -d
```

Cek status layanan:
```bash
docker-compose ps
```

### 3. Setup MinIO Bucket

Setelah MinIO berjalan, buat bucket untuk media files:

1. Buka MinIO Console: http://localhost:9001
2. Login dengan credentials dari `.env` (default: minioadmin/minioadmin123)
3. Buat bucket baru dengan nama `fish-media` (atau sesuai `MINIO_BUCKET_NAME` di `.env`)
4. Set bucket policy menjadi public atau sesuai kebutuhan

Atau gunakan MinIO Client (mc):
```bash
# Install mc
brew install minio/stable/mc  # macOS
# atau download dari https://min.io/docs/minio/linux/reference/minio-mc.html

# Configure
mc alias set local http://localhost:9000 minioadmin minioadmin123

# Create bucket
mc mb local/fish-media

# Set policy (optional - for public access)
mc anonymous set download local/fish-media
```

### 4. Install Python Dependencies

```bash
cd fish_api
pip install -r requirements.txt
```

### 5. Migrate Database

```bash
python manage.py migrate
```

### 6. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 7. Run Django Application

```bash
daphne -b 0.0.0.0 -p 8000 fish_recognition_api.asgi:application
```

Atau dengan Daphne di port 8001:
```bash
daphne -b 0.0.0.0 -p 8001 fish_recognition_api.asgi:application
```

## Service URLs

- **Django API**: http://localhost:8000 (atau 8001)
- **MinIO Console**: http://localhost:9001
- **MinIO API**: http://localhost:9000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:36379

## Mengelola Docker Services

### Stop Services
```bash
docker-compose stop
```

### Start Services (yang sudah dibuat)
```bash
docker-compose start
```

### Restart Services
```bash
docker-compose restart
```

### Stop dan Hapus Containers
```bash
docker-compose down
```

### Stop dan Hapus Containers + Volumes (HATI-HATI: Data akan hilang!)
```bash
docker-compose down -v
```

### Lihat Logs
```bash
# Semua services
docker-compose logs -f

# Service tertentu
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f minio
```

## Troubleshooting

### PostgreSQL Connection Error

Pastikan PostgreSQL sudah running dan environment variables sudah benar:
```bash
docker-compose ps postgres
docker-compose logs postgres
```

Test koneksi:
```bash
docker exec -it fish_postgres psql -U fish_user -d fish_recognition
```

### MinIO Connection Error

Pastikan MinIO sudah running dan bucket sudah dibuat:
```bash
docker-compose ps minio
docker-compose logs minio
```

### Redis Connection Error

Pastikan Redis sudah running:
```bash
docker-compose ps redis
docker-compose logs redis
```

Test koneksi:
```bash
docker exec -it fish_redis redis-cli ping
```

## Switching Between SQLite and PostgreSQL

Aplikasi akan otomatis menggunakan PostgreSQL jika environment variable `POSTGRES_HOST` atau `DATABASE_URL` diset.

Untuk menggunakan SQLite (development lokal tanpa Docker):
```bash
# Hapus atau comment environment variables PostgreSQL di .env
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# dll
```

Untuk menggunakan PostgreSQL:
```bash
# Set environment variables PostgreSQL di .env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=fish_recognition
POSTGRES_USER=fish_user
POSTGRES_PASSWORD=fish_password
```

## Production Considerations

Untuk production deployment:

1. **Ganti Secret Keys**: Update `SECRET_KEY`, passwords, dan credentials
2. **Set DEBUG=False**: Disable debug mode
3. **Configure ALLOWED_HOSTS**: Set specific hosts, bukan `*`
4. **Use SSL for MinIO**: Set `MINIO_USE_SSL=True` dan gunakan HTTPS
5. **Backup Strategy**: Setup regular backups untuk PostgreSQL dan MinIO
6. **Security**: Review security settings di settings.py
7. **Performance**: Tune PostgreSQL dan Redis untuk production workload

## MinIO Storage Configuration

Django akan otomatis menggunakan MinIO untuk media files jika `MINIO_ENDPOINT` diset di environment variables.

Untuk upload files, gunakan Django's file upload API seperti biasa. Files akan otomatis disimpan di MinIO.

Contoh di views.py:
```python
from django.core.files.storage import default_storage

# Save file
file_path = default_storage.save('images/fish.jpg', file_content)

# Get URL
file_url = default_storage.url(file_path)
```

## Environment Variables Reference

Lihat `.env.example` untuk daftar lengkap environment variables yang tersedia.

## Support

Jika ada masalah, cek:
1. Docker logs: `docker-compose logs -f`
2. Django logs: Lihat console output atau file log
3. Pastikan semua services healthy: `docker-compose ps`
