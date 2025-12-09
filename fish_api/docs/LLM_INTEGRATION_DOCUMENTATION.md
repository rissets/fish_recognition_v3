# LLM Integration Documentation
## Integrasi Ollama gamma3 untuk Enhanced Fish Classification

### Overview
Integrasi ini menambahkan layer verifikasi menggunakan LLM (Large Language Model) Ollama gamma3 untuk meningkatkan akurasi klasifikasi ikan. Model gamma3 adalah vision model yang dapat menganalisis gambar dan memberikan identifikasi yang lebih akurat.

### Flow Sistem

```
Input Image
    ↓
[Fish Detection] ← Model YOLOv8/v10
    ↓
[Face Filtering] ← Filter deteksi wajah manusia
    ↓
[Fish Classification] ← Model Embedding Classifier (639 classes)
    ↓
[LLM Verification] ← Ollama gamma3 (NEW!)
    ↓
[Segmentation] ← Model Segmentasi
    ↓
Final Result (dengan LLM verification)
```

### Fitur LLM Enhancement

1. **Verifikasi Klasifikasi**
   - LLM menerima gambar cropped dari ikan yang terdeteksi
   - Menerima hasil prediksi dari model klasifikasi
   - Memberikan verifikasi nama ilmiah dan nama Indonesia

2. **Output Format**
   ```json
   {
     "scientific_name": "Oreochromis mossambicus",
     "indonesian_name": "Ikan Mujair"
   }
   ```

3. **System Prompt**
   LLM dikonfigurasi dengan prompt khusus untuk:
   - Fokus pada identifikasi spesies ikan
   - Output JSON struktural
   - Memberikan "Unknown" jika tidak dapat mengidentifikasi

### API Endpoints

#### 1. Recognition Endpoint (Updated)
**POST** `/api/recognition/recognize/`

Response sekarang includes `llm_verification`:
```json
{
  "success": true,
  "fish_detections": [
    {
      "id": 0,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "classification": [
        {
          "label": "Oreochromis mossambicus",
          "confidence": 0.85,
          "class_id": 450
        }
      ],
      "llm_verification": {
        "scientific_name": "Oreochromis mossambicus",
        "indonesian_name": "Ikan Mujair",
        "processing_time": 2.5
      },
      "segmentation": {...}
    }
  ]
}
```

#### 2. LLM Configuration Endpoint (NEW)
**GET** `/api/recognition/config/llm/`

Get current LLM configuration:
```json
{
  "enabled": true,
  "service_available": true,
  "health": {
    "status": "healthy",
    "url": "https://ollama.hellodigi.id",
    "model": "gamma3",
    "model_available": true
  },
  "description": "LLM enhancement uses Ollama gamma3 model for improved classification accuracy"
}
```

**POST** `/api/recognition/config/llm/`

Update LLM configuration:
```json
{
  "enabled": true
}
```

Response:
```json
{
  "message": "LLM configuration updated successfully",
  "config": {
    "enabled": true,
    "service_available": true,
    "health": {...}
  }
}
```

#### 3. Health Check Endpoint (Updated)
**GET** `/api/recognition/health/`

Response now includes LLM status:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "llm_enhancement": {
    "enabled": true,
    "service_available": true,
    "health": {
      "status": "healthy",
      "model": "gamma3"
    }
  }
}
```

### Environment Variables

Tambahkan ke `.env`:
```bash
# Ollama LLM Settings
OLLAMA_URL=https://ollama.hellodigi.id
OLLAMA_MODEL=gamma3
LLM_ENABLED=True
LLM_TIMEOUT=30
```

### Architecture

#### 1. OllamaLLMService
File: `recognition/services/ollama_llm_service.py`

**Methods:**
- `verify_classification(image, detection_info)` - Verifikasi single image
- `batch_verify(images, detection_infos)` - Batch verification
- `health_check()` - Check service availability

#### 2. FishRecognitionEngine (Updated)
File: `recognition/ml_models/fish_engine.py`

**New Attributes:**
- `llm_enabled`: Boolean untuk enable/disable LLM
- `llm_service`: Instance dari OllamaLLMService

**New Methods:**
- `_initialize_llm()`: Initialize LLM service
- `configure_llm(enabled)`: Configure LLM settings
- `get_llm_config()`: Get current LLM configuration

**Updated Methods:**
- `process_image()`: Includes LLM verification step
- `health_check()`: Includes LLM health status

### Performance Considerations

1. **LLM Processing Time**
   - Rata-rata: 2-5 detik per image
   - Timeout: 30 detik (configurable)
   - Tidak blocking jika LLM gagal

2. **Fallback Behavior**
   - Jika LLM gagal/timeout: sistem tetap return hasil klasifikasi normal
   - `llm_verification` akan berisi `null` atau error message

3. **Statistics Tracking**
   - Processing time LLM tercatat di `processing_stats['llm_verification']`
   - Available di `/api/recognition/stats/` endpoint

### Testing

Test script tersedia di `test_llm_integration.py`:

```bash
cd fish_api
python test_llm_integration.py
```

Test scenarios:
1. LLM health check
2. Single image recognition with LLM
3. LLM configuration toggle
4. Batch processing with LLM

### Error Handling

1. **LLM Service Unavailable**
   ```json
   {
     "llm_verification": null,
     "warning": "LLM service unavailable"
   }
   ```

2. **LLM Timeout**
   ```json
   {
     "llm_verification": {
       "error": "Request timeout"
     }
   }
   ```

3. **Invalid Response**
   ```json
   {
     "llm_verification": {
       "error": "Invalid JSON response from LLM"
     }
   }
   ```

### Best Practices

1. **Enable LLM for Critical Classifications**
   - Gunakan LLM untuk spesies yang sulit dibedakan
   - Non-critical: bisa disable untuk performance

2. **Monitor Performance**
   - Check LLM processing time regularly
   - Monitor timeout rate

3. **Fallback Strategy**
   - Selalu ada hasil dari model klasifikasi
   - LLM sebagai enhancement, bukan replacement

### Future Improvements

1. **Caching**
   - Cache hasil LLM untuk gambar similar
   - Reduce redundant LLM calls

2. **Confidence Scoring**
   - Compare LLM result dengan model prediction
   - Generate combined confidence score

3. **Multi-Model Ensemble**
   - Combine multiple LLM responses
   - Vote-based final decision

### Troubleshooting

#### LLM Service Tidak Tersedia
```bash
# Check health endpoint
curl http://localhost:8001/api/recognition/config/llm/
```

#### Slow Response
- Check `LLM_TIMEOUT` setting
- Monitor network latency ke Ollama server

#### Inconsistent Results
- Verify system prompt
- Check model availability
- Review LLM response logs

### Support

Untuk pertanyaan atau issue, check:
1. API logs: `fish_api/fish_api.log`
2. Health endpoint: `/api/recognition/health/`
3. LLM config: `/api/recognition/config/llm/`
