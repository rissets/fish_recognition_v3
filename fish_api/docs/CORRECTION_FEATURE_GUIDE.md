# Panduan Fitur Koreksi Identifikasi Ikan

## Ringkasan
Fitur ini memungkinkan pengguna mengkoreksi hasil identifikasi ikan yang salah dari AI melalui interface web atau REST API.

## Cara Menggunakan di Web Interface

### Langkah-langkah:

1. **Upload dan Identifikasi Ikan**
   - Upload gambar ikan melalui mode Image Upload atau Camera
   - Tunggu hasil identifikasi dari AI
   - Hasil akan muncul di panel "Recent Results"

2. **Membuka Form Koreksi**
   - Pada result card, klik tombol **"🔧 Koreksi Identifikasi"** (berwarna orange)
   - Modal koreksi akan muncul dengan data saat ini sudah terisi

3. **Mengisi Form Koreksi**
   - **Nama Ilmiah** (wajib): Contoh `Oreochromis mossambicus`
   - **Nama Indonesia** (wajib): Contoh `Ikan Mujair`
   - **Nama Inggris** (opsional): Contoh `Mozambique Tilapia`
   - **Kelompok** (opsional): Contoh `Ikan Air Tawar`
   - **Catatan Koreksi** (opsional): Jelaskan alasan koreksi

4. **Submit Koreksi**
   - Klik tombol **"Simpan Koreksi"**
   - Sistem akan memproses dan menyimpan koreksi
   - Result card akan diupdate dengan nama yang benar
   - Badge "✓ Dikoreksi" akan muncul

## Cara Menggunakan REST API

### Endpoint
```
POST /api/v1/identifications/<identification_id>/correct/
```

### Request Body (JSON)
```json
{
    "scientific_name": "Oreochromis mossambicus",
    "indonesian_name": "Ikan Mujair",
    "english_name": "Mozambique Tilapia",
    "kelompok": "Ikan Air Tawar",
    "notes": "Koreksi berdasarkan verifikasi visual"
}
```

### Contoh dengan cURL
```bash
curl -X POST http://localhost:8000/api/v1/identifications/YOUR_ID_HERE/correct/ \
  -H "Content-Type: application/json" \
  -d '{
    "scientific_name": "Channa striata",
    "indonesian_name": "Ikan Gabus",
    "notes": "Koreksi manual"
  }'
```

### Contoh dengan Python
```python
import requests

identification_id = "550e8400-e29b-41d4-a716-446655440000"

correction = {
    "scientific_name": "Channa striata",
    "indonesian_name": "Ikan Gabus",
    "english_name": "Striped Snakehead",
    "notes": "Identifikasi awal kurang tepat"
}

response = requests.post(
    f"http://localhost:8000/api/v1/identifications/{identification_id}/correct/",
    json=correction
)

print(response.json())
```

### Contoh dengan JavaScript/Fetch
```javascript
const identificationId = "550e8400-e29b-41d4-a716-446655440000";

const correction = {
    scientific_name: "Channa striata",
    indonesian_name: "Ikan Gabus",
    english_name: "Striped Snakehead",
    notes: "Koreksi setelah konsultasi ahli"
};

fetch(`/api/v1/identifications/${identificationId}/correct/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(correction)
})
.then(res => res.json())
.then(data => console.log('Corrected:', data))
.catch(err => console.error('Error:', err));
```

## Response API

### Success (200 OK)
```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "current_scientific_name": "Oreochromis mossambicus",
    "current_indonesian_name": "Ikan Mujair",
    "is_corrected": true,
    "corrected_at": "2024-12-09T10:30:00Z",
    "status": "corrected"
}
```

### Error (404 Not Found)
```json
{
    "error": "Fish identification not found"
}
```

### Error (400 Bad Request)
```json
{
    "scientific_name": ["This field is required"],
    "indonesian_name": ["This field is required"]
}
```

## Field Requirements

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scientific_name` | string | ✅ Yes | Nama ilmiah (Latin) |
| `indonesian_name` | string | ✅ Yes | Nama Indonesia |
| `english_name` | string | ❌ No | Nama Inggris |
| `kelompok` | string | ❌ No | Kelompok ikan |
| `notes` | string | ❌ No | Catatan/alasan koreksi |

## Data yang Tersimpan

Sistem menyimpan:

1. **Original Data** - Data prediksi awal dari AI
   - `original_scientific_name`
   - `original_indonesian_name`
   - `original_english_name`

2. **Current Data** - Data terbaru (hasil koreksi)
   - `current_scientific_name`
   - `current_indonesian_name`
   - `current_english_name`
   - `current_kelompok`

3. **Metadata Koreksi**
   - `is_corrected` - Boolean flag
   - `corrected_at` - Timestamp koreksi
   - `correction_notes` - Catatan koreksi
   - `status` - Status: 'corrected'

4. **History** - Riwayat perubahan lengkap di tabel terpisah

## Tips & Best Practices

### ✅ Do's
- Gunakan nama ilmiah yang valid (format: *Genus species*)
- Berikan catatan koreksi untuk tracking yang lebih baik
- Verifikasi kembali sebelum submit
- Gunakan referensi reliable untuk nama ikan

### ❌ Don'ts
- Jangan gunakan singkatan untuk nama ilmiah
- Jangan lupa fill required fields
- Jangan submit tanpa verifikasi data

## FAQ

**Q: Apakah bisa koreksi berkali-kali?**
A: Ya, bisa. Setiap koreksi akan tercatat dalam history.

**Q: Apakah data original hilang setelah koreksi?**
A: Tidak. Data original tetap tersimpan di field `original_*`.

**Q: Bagaimana cara melihat history koreksi?**
A: History tersimpan di tabel `FishIdentificationHistory` (endpoint akan ditambahkan).

**Q: Apakah koreksi mempengaruhi model AI?**
A: Saat ini belum, tapi data koreksi bisa digunakan untuk melatih ulang model di masa depan.

**Q: Identification ID dari mana?**
A: Setiap hasil identifikasi dari endpoint `/api/v1/recognize/` akan return `identification_id`.

## Dokumentasi Lengkap

Untuk dokumentasi API yang lebih detail, lihat:
- `/fish_api/docs/CORRECTION_API.md` - Dokumentasi lengkap API
- `/fish_api/FISH_IDENTIFICATION_API.md` - Main API documentation

## Troubleshooting

### Problem: Tombol koreksi tidak muncul
**Solution:** Pastikan hasil identifikasi memiliki `identification_id` dan ada fish detections

### Problem: Error 404 saat submit
**Solution:** Cek apakah identification ID valid dan exists di database

### Problem: Error 400 saat submit
**Solution:** Pastikan `scientific_name` dan `indonesian_name` terisi

### Problem: Modal tidak muncul
**Solution:** Cek console browser untuk error JavaScript, reload page

## Support

Jika ada pertanyaan atau masalah:
1. Cek dokumentasi lengkap di `/fish_api/docs/`
2. Periksa console browser untuk error
3. Check server logs di terminal
4. Buka GitHub Issues untuk bug report
