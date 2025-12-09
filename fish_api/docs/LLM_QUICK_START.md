# LLM Integration - Quick Start Guide

## 🚀 Pengenalan

Integrasi ini menambahkan **Ollama gamma3** (vision model) untuk meningkatkan akurasi klasifikasi ikan. LLM digunakan sebagai layer verifikasi tambahan setelah model klasifikasi utama.

### Keuntungan
- ✅ Akurasi lebih tinggi dengan verifikasi LLM
- ✅ Nama ilmiah dan nama Indonesia yang lebih akurat
- ✅ Fallback ke model klasifikasi jika LLM gagal
- ✅ Dapat di-enable/disable secara dinamis
- ✅ Tidak mengganggu flow existing

## 📋 Prerequisites

- Python 3.8+
- Django 4.2+
- Ollama server running (https://ollama.hellodigi.id)
- Model gamma3 tersedia di Ollama server

## 🔧 Setup

### 1. Update Environment Variables

Edit file `.env` atau buat dari `.env.example`:

```bash
# Ollama LLM Settings
OLLAMA_URL=https://ollama.hellodigi.id
OLLAMA_MODEL=gamma3
LLM_ENABLED=True
LLM_TIMEOUT=30
```

### 2. Install Dependencies

Dependencies sudah ada di `requirements.txt` (requests sudah included).

```bash
cd fish_api
pip install -r requirements.txt
```

### 3. Restart Server

```bash
cd fish_api
python manage.py runserver 0.0.0.0:8001

# Atau dengan Daphne (untuk WebSocket support)
daphne -b 0.0.0.0 -p 8001 fish_recognition_api.asgi:application
```

## 🧪 Testing

### Quick Health Check

```bash
# Check LLM status
curl http://localhost:8001/api/recognition/config/llm/
```

Expected response:
```json
{
  "enabled": true,
  "service_available": true,
  "health": {
    "status": "healthy",
    "url": "https://ollama.hellodigi.id",
    "model": "gamma3",
    "model_available": true
  }
}
```

### Run Test Suite

```bash
cd fish_api
python test_llm_integration.py

# With test image
python test_llm_integration.py --image /path/to/fish/image.jpg

# Specific test
python test_llm_integration.py --test health
```

## 📖 Usage

### Single Image Recognition

```python
import requests

# Upload image
with open('fish_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post(
        'http://localhost:8001/api/recognition/recognize/',
        files=files
    )

result = response.json()

# Check results
for fish in result['fish_detections']:
    print(f"Classification: {fish['classification'][0]['label']}")
    
    # LLM verification (new!)
    if fish['llm_verification']:
        llm = fish['llm_verification']
        print(f"LLM Scientific: {llm['scientific_name']}")
        print(f"LLM Indonesian: {llm['indonesian_name']}")
```

### Using cURL

```bash
curl -X POST http://localhost:8001/api/recognition/recognize/ \
  -F "image=@fish_image.jpg" \
  -F "include_faces=true" \
  -F "include_segmentation=true"
```

### Enable/Disable LLM

```bash
# Disable LLM
curl -X POST http://localhost:8001/api/recognition/config/llm/ \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# Enable LLM
curl -X POST http://localhost:8001/api/recognition/config/llm/ \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

## 📊 Response Format

### With LLM Verification

```json
{
  "success": true,
  "fish_detections": [
    {
      "id": 0,
      "bbox": [100, 200, 300, 400],
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
      "segmentation": {
        "has_segmentation": true,
        "polygon_data": [...]
      }
    }
  ],
  "total_processing_time": 3.2
}
```

### LLM Disabled or Failed

```json
{
  "llm_verification": null
}
```

## 🔍 Monitoring

### Check System Health

```bash
curl http://localhost:8001/api/recognition/health/
```

### Performance Statistics

```bash
curl http://localhost:8001/api/recognition/stats/
```

Response includes LLM processing times:
```json
{
  "llm_verification": {
    "count": 10,
    "mean": 2.5,
    "median": 2.3,
    "min": 1.8,
    "max": 4.2
  }
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `https://ollama.hellodigi.id` | Ollama API URL |
| `OLLAMA_MODEL` | `gamma3` | Model name |
| `LLM_ENABLED` | `True` | Enable/disable LLM |
| `LLM_TIMEOUT` | `30` | Request timeout (seconds) |

### Runtime Configuration

LLM can be enabled/disabled at runtime via API:

```bash
POST /api/recognition/config/llm/
{"enabled": true/false}
```

No server restart required!

## 🐛 Troubleshooting

### LLM Service Unavailable

**Problem**: `"service_available": false`

**Solutions**:
1. Check Ollama server: `curl https://ollama.hellodigi.id/api/tags`
2. Verify network connectivity
3. Check firewall settings
4. Review logs: `tail -f fish_api/fish_api.log`

### Slow Response Times

**Problem**: Recognition takes too long

**Solutions**:
1. Check `LLM_TIMEOUT` setting
2. Disable LLM for non-critical use: `{"enabled": false}`
3. Monitor network latency to Ollama server
4. Consider caching strategies

### Inconsistent Results

**Problem**: LLM returns different results than classification

**Expected**: This is normal! LLM provides additional verification.

**Actions**:
1. Compare both results
2. Use LLM as tiebreaker for low-confidence classifications
3. Log discrepancies for model improvement

### Import Error

**Problem**: `ModuleNotFoundError: No module named 'recognition.services.ollama_llm_service'`

**Solution**:
```bash
# Restart Django server
cd fish_api
python manage.py runserver 0.0.0.0:8001
```

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/recognition/recognize/` | Single image recognition (with LLM) |
| POST | `/api/recognition/recognize/batch/` | Batch image recognition (with LLM) |
| GET | `/api/recognition/config/llm/` | Get LLM configuration |
| POST | `/api/recognition/config/llm/` | Update LLM configuration |
| GET | `/api/recognition/health/` | System health (includes LLM status) |
| GET | `/api/recognition/stats/` | Performance statistics |

## 📚 Documentation

- **Full Documentation**: `LLM_INTEGRATION_DOCUMENTATION.md`
- **Flow Diagram**: `LLM_FLOW_DIAGRAM.md`
- **Test Script**: `test_llm_integration.py`

## 🎯 Best Practices

### For Production

1. **Monitor Performance**
   ```bash
   # Regular health checks
   */5 * * * * curl http://localhost:8001/api/recognition/config/llm/
   ```

2. **Set Appropriate Timeout**
   - Production: 30s (default)
   - Development: 60s
   - High-volume: 15s

3. **Error Handling**
   - Always check `llm_verification` is not null before using
   - Have fallback to `classification` results

4. **Load Balancing**
   - Consider LLM call overhead
   - Use selective LLM (only for low confidence)

### For Development

1. **Quick Testing**
   ```bash
   # Disable LLM for faster iteration
   curl -X POST http://localhost:8001/api/recognition/config/llm/ \
     -H "Content-Type: application/json" \
     -d '{"enabled": false}'
   ```

2. **Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## 🆘 Support

Jika ada masalah:

1. Check logs: `fish_api/fish_api.log`
2. Run test suite: `python test_llm_integration.py`
3. Check health endpoint: `/api/recognition/health/`
4. Review documentation: `LLM_INTEGRATION_DOCUMENTATION.md`

## 🎉 Success Checklist

- [ ] Environment variables configured
- [ ] Server restarted
- [ ] Health check passes
- [ ] Test suite passes
- [ ] LLM verification appears in responses
- [ ] Performance acceptable

## 📈 Next Steps

1. Monitor LLM accuracy vs classification model
2. Collect metrics on processing time
3. Implement caching for repeated images
4. Add confidence-based LLM triggering
5. Consider ensemble methods for final decision

---

**Need Help?** Check the full documentation or run `python test_llm_integration.py --help`
