# API Documentation dengan Swagger/OpenAPI

## Overview

Fish Recognition API sekarang dilengkapi dengan dokumentasi Swagger/OpenAPI yang komprehensif menggunakan **drf-spectacular**. Dokumentasi ini menyediakan:

- ✅ Daftar lengkap semua endpoint API
- ✅ Request & Response schema yang detail
- ✅ Contoh request/response untuk setiap endpoint
- ✅ Interactive testing interface (Try it out)
- ✅ Pengelompokan endpoint berdasarkan kategori

## Akses Swagger UI

### 1. Swagger UI (Interaktif)
```
http://localhost:8001/api/docs/
```

**Fitur:**
- Interactive API explorer
- Try out requests langsung dari browser
- Lihat real-time responses
- Auto-complete untuk parameter

### 2. ReDoc (Dokumentasi yang Lebih Rapi)
```
http://localhost:8001/api/redoc/
```

**Fitur:**
- Dokumentasi yang lebih terorganisir
- Layout yang lebih clean untuk reading
- Lebih cocok untuk dokumentasi tim

### 3. OpenAPI Schema (Raw)
```
http://localhost:8001/api/schema/
```

**Format:** JSON/YAML
**Kegunaan:** Import ke tools lain (Postman, Insomnia, dll)

## Kategori Endpoint

### 🔍 Recognition
Endpoint utama untuk fish detection dan recognition
- `POST /api/v1/recognize/` - Single image recognition
- `POST /api/v1/recognize/batch/` - Batch image recognition

### ❤️ Health
Health check dan monitoring
- `GET /api/v1/health/` - API health status
- `GET /api/v1/stats/` - Performance statistics
- `GET /api/v1/minio/health/` - MinIO storage health

### ⚙️ Configuration
Model dan system configuration
- `GET/POST /api/v1/config/` - Model configuration
- `GET/POST /api/v1/config/face-filter/` - Face filter settings
- `POST /api/v1/embedding/lwf/` - LWF adaptation

### 🐟 Identification
Fish identification management
- `GET /api/v1/identifications/` - List identifications
- `GET /api/v1/identifications/{id}/` - Get details
- `GET /api/v1/identifications/{id}/history/` - Change history
- `GET /api/v1/species/statistics/` - Species statistics

### ✏️ Correction
Manual correction dan verification
- `POST /api/v1/identifications/{id}/correct/` - Correct identification
- `POST /api/v1/identifications/{id}/verify/` - Verify correct
- `POST /api/v1/identifications/{id}/reject/` - Reject identification

### 📊 Dataset
Dataset management dan export
- `GET /api/v1/dataset/export/` - Export dataset
- `GET /api/v1/dataset/statistics/` - Dataset stats
- `GET /api/v1/master-data/` - Fish master data
- `GET /api/v1/master-data/stats/` - Master data statistics

## Fitur Dokumentasi

### 1. Request Examples
Setiap endpoint dilengkapi dengan contoh request:
- Multipart file upload
- Base64 image encoding
- JSON body examples
- Query parameter examples

### 2. Response Examples
Response examples untuk berbagai scenario:
- ✅ Success responses
- ❌ Error responses  
- ⚠️ Validation errors
- 🚫 Not found errors

### 3. Schema Details
Dokumentasi lengkap untuk:
- Field types dan validations
- Required vs optional fields
- Default values
- Min/max values
- Enum choices

### 4. Authentication
Saat ini API menggunakan `AllowAny` permission untuk development.
Untuk production, akan ditambahkan authentication yang didokumentasikan di Swagger.

## Generate Schema Manual

Jika perlu regenerate schema file:

```bash
cd fish_api
python manage.py spectacular --color --file schema.yml
```

## Import ke Tools Lain

### Postman
1. Buka Postman
2. File → Import
3. Pilih "Link" dan paste: `http://localhost:8001/api/schema/`
4. Import collection

### Insomnia
1. Buka Insomnia
2. Create → Import from URL
3. Paste: `http://localhost:8001/api/schema/`
4. Import

### VS Code REST Client
Download schema dan gunakan dengan OpenAPI extension

## Tips Penggunaan

### Try It Out di Swagger UI

1. **Buka endpoint** yang ingin dicoba
2. **Klik "Try it out"**
3. **Isi parameter** atau upload file
4. **Klik "Execute"**
5. **Lihat response** di bagian bawah

### Testing Image Recognition

**Single Image:**
```bash
# Via Swagger UI:
1. POST /api/v1/recognize/
2. Upload image file atau paste base64
3. Set include_visualization = true
4. Execute

# Atau via curl:
curl -X POST http://localhost:8001/api/v1/recognize/ \
  -F "image=@path/to/fish.jpg" \
  -F "include_visualization=true"
```

**Batch Images:**
```bash
# Via Swagger UI:
1. POST /api/v1/recognize/batch/
2. Upload multiple images (max 10)
3. Execute
```

### Filtering & Search

Banyak endpoint support filtering:

**Identifications:**
```
GET /api/v1/identifications/?status=verified&search=tuna
```

**Master Data:**
```
GET /api/v1/master-data/?kelompok=Ikan%20Pelagis&search=tuna
```

**Statistics:**
```
GET /api/v1/species/statistics/?min_identifications=10&kelompok=Ikan%20Pelagis
```

## Update Dokumentasi

Setelah menambah/mengubah endpoint:

1. **Tambahkan `@extend_schema` decorator**
   ```python
   @extend_schema(
       tags=['Category'],
       summary='Short description',
       description='Detailed description',
       request=RequestSerializer,
       responses={200: ResponseSerializer}
   )
   class MyView(APIView):
       ...
   ```

2. **Update serializer dengan help_text**
   ```python
   field = serializers.CharField(
       help_text="Description of this field"
   )
   ```

3. **Tambahkan examples di `schema_examples.py`**
   ```python
   MY_EXAMPLE_RESPONSE = {
       "key": "value"
   }
   ```

4. **Regenerate schema** (opsional)
   ```bash
   python manage.py spectacular --file schema.yml
   ```

## Troubleshooting

### Schema tidak ter-update
```bash
# Clear cache dan restart
rm -f schema.yml
python manage.py spectacular --file schema.yml
# Restart Django server
```

### Warning saat generate
Warnings seperti "could not resolve OpenApiExample" adalah normal dan tidak mempengaruhi fungsionalitas.

### Error 404 saat akses /api/docs/
Pastikan:
1. `drf-spectacular` sudah ter-install
2. URLs sudah di-include di `urls.py`
3. Django server running

## Resources

- **drf-spectacular docs:** https://drf-spectacular.readthedocs.io/
- **OpenAPI Spec:** https://swagger.io/specification/
- **Swagger UI:** https://swagger.io/tools/swagger-ui/

## Support

Jika ada pertanyaan atau issue dengan dokumentasi API:
1. Check Swagger UI untuk contoh request/response
2. Lihat `schema_examples.py` untuk reference
3. Regenerate schema jika perlu
