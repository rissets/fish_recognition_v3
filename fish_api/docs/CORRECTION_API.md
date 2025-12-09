# Fish Identification Correction API

## Overview
Fitur koreksi identifikasi memungkinkan pengguna untuk memperbaiki hasil identifikasi ikan yang salah dari AI. Sistem akan menyimpan history perubahan dan menandai identifikasi yang telah dikoreksi.

## Endpoint API

### Koreksi Identifikasi
**URL:** `/api/v1/identifications/<identification_id>/correct/`  
**Method:** `POST`  
**Content-Type:** `application/json`

#### Request Body
```json
{
    "scientific_name": "Oreochromis mossambicus",
    "indonesian_name": "Ikan Mujair",
    "english_name": "Mozambique Tilapia",
    "kelompok": "Ikan Air Tawar",
    "notes": "Koreksi berdasarkan verifikasi visual detail sirip dan corak tubuh"
}
```

#### Request Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scientific_name` | string | Yes | Nama ilmiah yang benar |
| `indonesian_name` | string | Yes | Nama Indonesia yang benar |
| `english_name` | string | No | Nama Inggris (opsional) |
| `kelompok` | string | No | Kelompok ikan (opsional) |
| `notes` | string | No | Catatan atau alasan koreksi |

#### Response Success (200 OK)
```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "current_scientific_name": "Oreochromis mossambicus",
    "current_indonesian_name": "Ikan Mujair",
    "current_english_name": "Mozambique Tilapia",
    "current_kelompok": "Ikan Air Tawar",
    "is_corrected": true,
    "corrected_at": "2024-12-09T10:30:00Z",
    "correction_notes": "Koreksi berdasarkan verifikasi visual detail sirip dan corak tubuh",
    "original_scientific_name": "Oreochromis niloticus",
    "original_indonesian_name": "Ikan Nila",
    "status": "corrected"
}
```

#### Response Error (404 Not Found)
```json
{
    "error": "Fish identification not found"
}
```

#### Response Error (400 Bad Request)
```json
{
    "scientific_name": ["This field is required"],
    "indonesian_name": ["This field is required"]
}
```

## Frontend Integration

### HTML Modal Form
Modal koreksi sudah tersedia di `templates/index.html` dengan form yang lengkap:
- Input nama ilmiah (required)
- Input nama Indonesia (required)
- Input nama Inggris (optional)
- Input kelompok (optional)
- Textarea catatan koreksi (optional)

### JavaScript Functions

#### Membuka Modal Koreksi
```javascript
fishApp.showCorrectionModal(identificationId, currentData);
```

**Parameters:**
- `identificationId` (string): UUID dari identifikasi yang akan dikoreksi
- `currentData` (object): Data saat ini untuk pre-fill form
  ```javascript
  {
      scientific_name: "...",
      indonesian_name: "...",
      english_name: "...",
      kelompok: "..."
  }
  ```

#### Submit Koreksi
Form akan otomatis mengirim POST request ke endpoint API ketika user menekan tombol "Simpan Koreksi".

#### Update UI Setelah Koreksi
Setelah koreksi berhasil, UI akan otomatis:
1. Update nama spesies di result card
2. Menambahkan badge "✓ Dikoreksi"
3. Menutup modal
4. Menampilkan notifikasi sukses

## Usage Examples

### Contoh 1: Koreksi dari JavaScript
```javascript
const correctionData = {
    scientific_name: "Channa striata",
    indonesian_name: "Ikan Gabus",
    english_name: "Striped Snakehead",
    kelompok: "Ikan Predator",
    notes: "Identifikasi awal salah, ini adalah ikan gabus bukan kutuk"
};

const response = await fetch(`/api/v1/identifications/${identificationId}/correct/`, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(correctionData)
});

const result = await response.json();
console.log('Correction successful:', result);
```

### Contoh 2: Koreksi dari cURL
```bash
curl -X POST http://localhost:8000/api/v1/identifications/550e8400-e29b-41d4-a716-446655440000/correct/ \
  -H "Content-Type: application/json" \
  -d '{
    "scientific_name": "Channa striata",
    "indonesian_name": "Ikan Gabus",
    "english_name": "Striped Snakehead",
    "notes": "Koreksi manual setelah verifikasi ahli"
  }'
```

### Contoh 3: Koreksi dari Python
```python
import requests

identification_id = "550e8400-e29b-41d4-a716-446655440000"
correction_data = {
    "scientific_name": "Channa striata",
    "indonesian_name": "Ikan Gabus",
    "english_name": "Striped Snakehead",
    "kelompok": "Ikan Predator",
    "notes": "Koreksi berdasarkan konsultasi dengan ahli iktiologi"
}

response = requests.post(
    f"http://localhost:8000/api/v1/identifications/{identification_id}/correct/",
    json=correction_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Corrected to: {result['current_indonesian_name']}")
else:
    print(f"Error: {response.json()}")
```

## Database Schema

### FishIdentification Model Fields
```python
# Original AI prediction
original_scientific_name = CharField()
original_indonesian_name = CharField()
original_english_name = CharField()

# Current (possibly corrected) identification
current_scientific_name = CharField()
current_indonesian_name = CharField()
current_english_name = CharField()
current_kelompok = CharField()

# Correction metadata
is_corrected = BooleanField()
corrected_at = DateTimeField()
correction_notes = TextField()
status = CharField()  # 'pending', 'verified', 'corrected', 'flagged'
```

### FishIdentificationHistory Model
Setiap perubahan dicatat dalam history table:
```python
identification = ForeignKey(FishIdentification)
field_name = CharField()  # 'scientific_name', 'indonesian_name', etc.
old_value = TextField()
new_value = TextField()
changed_by = CharField()
change_reason = TextField()
changed_at = DateTimeField()
```

## Best Practices

### 1. Validasi Input
- Selalu isi minimal `scientific_name` dan `indonesian_name`
- Gunakan nama ilmiah yang valid (genus + species)
- Berikan catatan yang jelas untuk membantu tracking

### 2. Workflow Koreksi
1. User melihat hasil identifikasi dari AI
2. Jika tidak sesuai, klik tombol "🔧 Koreksi Identifikasi"
3. Form muncul dengan data saat ini sudah terisi
4. User mengubah nama yang salah
5. Tambahkan catatan koreksi (recommended)
6. Submit form
7. System menyimpan koreksi dan history

### 3. Tracking & Analytics
- Monitor `is_corrected` field untuk melihat akurasi AI
- Gunakan `FishIdentificationHistory` untuk audit trail
- Analisis correction patterns untuk improve model

## UI Features

### Result Card
Setiap result card yang memiliki `identification_id` akan menampilkan:
- Tombol "🔧 Koreksi Identifikasi" di bagian bawah card
- Badge "✓ Dikoreksi" setelah koreksi berhasil
- Update nama spesies secara real-time

### Modal Koreksi
- Pre-filled dengan data saat ini
- Validation untuk required fields
- Loading state saat submit
- Success/error notification
- Auto-close setelah sukses

## Error Handling

### Common Errors
1. **404 Not Found**: Identification ID tidak ditemukan
   - Cek apakah ID valid
   - Pastikan identifikasi exists di database

2. **400 Bad Request**: Validation error
   - Periksa required fields
   - Pastikan format data benar

3. **500 Server Error**: Internal server error
   - Check server logs
   - Verify database connection

### Error Messages
```javascript
try {
    await submitCorrection(data);
} catch (error) {
    if (error.response?.status === 404) {
        console.error('Identification not found');
    } else if (error.response?.status === 400) {
        console.error('Validation error:', error.response.data);
    } else {
        console.error('Server error:', error.message);
    }
}
```

## Testing

### Manual Testing
1. Upload gambar ikan
2. Tunggu hasil identifikasi
3. Klik tombol "Koreksi Identifikasi"
4. Ubah nama yang salah
5. Tambahkan catatan
6. Submit dan verify hasilnya

### API Testing with Postman
```json
POST /api/v1/identifications/{{identification_id}}/correct/
Content-Type: application/json

{
    "scientific_name": "Channa striata",
    "indonesian_name": "Ikan Gabus",
    "notes": "Test correction"
}
```

## Future Enhancements

### Planned Features
1. **Bulk Correction**: Koreksi multiple identifikasi sekaligus
2. **Correction Voting**: Multiple users vote on corrections
3. **Expert Review**: Flag untuk review oleh ahli
4. **Auto-learn**: Use corrections to retrain model
5. **Correction Statistics**: Dashboard untuk tracking correction rate

### API Extensions
- `GET /api/v1/identifications/<id>/history/` - View correction history
- `POST /api/v1/identifications/<id>/verify/` - Mark as verified (correct)
- `POST /api/v1/identifications/<id>/flag/` - Flag for review
- `GET /api/v1/corrections/statistics/` - Correction analytics

## Support & Documentation

### Related Endpoints
- `POST /api/v1/recognize/` - Fish recognition (returns identification_id)
- `GET /api/v1/identifications/<id>/` - Get identification details
- `GET /api/v1/history/` - Get identification history

### Contact
For issues or questions about the correction API, please refer to:
- Main API documentation: `FISH_IDENTIFICATION_API.md`
- Knowledge base integration: `KNOWLEDGE_BASE_INTEGRATION.md`
- GitHub Issues: (your repository URL)
