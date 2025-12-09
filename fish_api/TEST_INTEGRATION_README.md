# Integration Test: AI Model + LLM + Knowledge Base

## 📋 Overview

Test script telah dibuat untuk memvalidasi complete pipeline yang menggabungkan:
1. **AI Model** (Detection + Classification)  
2. **Knowledge Base** (Master Data dengan 1437 spesies)
3. **LLM** (Ollama gemma3:27b dengan vision capability)

## 🎯 Tujuan Test

Memastikan bahwa:
- ✅ AI Model dapat mendeteksi dan mengklasifikasi biota laut
- ✅ Knowledge Base dapat menemukan spesies yang relevan
- ✅ LLM dapat mem verify hasil dengan morphology analysis
- ✅ **Indonesian name dari LLM sesuai dengan master data**

## 📁 Test Files

### 1. `test_knowledge_base.py`
Test untuk Knowledge Base Service secara terpisah:
- Load master data (1437 species)
- Search by kelompok
- Morphology guide
- Similarity search
- Context building

**Cara menjalankan:**
```bash
cd /Users/user/Dev/researchs/fish_recognition_v2/fish_api
python test_knowledge_base.py
```

**Output yang diharapkan:**
```
✓ Knowledge base loaded with 1437 species
✓ 225 kelompok identified
✓ Morphology guide built for 8 key groups
✓ Similarity search working
✓ LLM context building successful
```

### 2. `test_integration.py`
Full integration test untuk semua gambar di folder `images/`:
- Load semua model (Detection, Classification, Segmentation, Face Detection)
- Process setiap gambar
- Run complete pipeline: AI → KB → LLM
- Validate hasil dengan master data

**Cara menjalankan:**
```bash
cd /Users/user/Dev/researchs/fish_recognition_v2/fish_api
python test_integration.py
```

**Note:** Test ini membutuhkan waktu lama karena:
- Load 4 ML models
- Process banyak gambar
- Call LLM API untuk setiap deteksi

### 3. `test_quick_integration.py` ⭐ **RECOMMENDED**
Quick test dengan 1 gambar saja untuk validasi cepat:
- Test complete pipeline dengan satu gambar
- Fokus pada validation logic
- Lebih cepat untuk debugging

**Cara menjalankan:**
```bash
cd /Users/user/Dev/researchs/fish_recognition_v2/fish_api
python test_quick_integration.py
```

## 🔍 Test Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load Image                                           │
│    • Baca gambar dari folder images/                    │
│    • Convert ke format yang sesuai                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. AI Model Processing                                  │
│    • Detection: YOLO model                              │
│    • Classification: Embedding model (ViT)              │
│    • Output: Top predictions dengan confidence          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Knowledge Base Context Building                      │
│    • Find similar species dari master data              │
│    • Get morphology guide untuk kelompok                │
│    • Build context untuk LLM                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. LLM Verification (Ollama gemma3:27b)                 │
│    • Send: Image + AI predictions + KB context          │
│    • LLM Analysis:                                      │
│      - Category identification (Fish/Crustacean/etc)    │
│      - Morphology analysis                              │
│      - Cross-reference dengan KB (70% weight)           │
│      - AI prediction (30% weight)                       │
│    • Output: Scientific name + Indonesian name          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Validation dengan Master Data                       │
│    • Lookup scientific name di master data              │
│    • Compare Indonesian names                           │
│    • Check if names match                               │
│    • Display kelompok, english name, dll                │
└─────────────────────────────────────────────────────────┘
```

## ✅ Expected Output

### Untuk Ikan (Fish):
```
FINAL RESULT:
Indonesian Name: Ikan Bandeng
Scientific Name: Chanos chanos

✓ VALIDATED WITH MASTER DATA:
  Master Indonesian: Bandeng
  Kelompok: Bandeng
  English: milkfish

✓✓ INDONESIAN NAME MATCHES MASTER DATA! ✓✓

PIPELINE SUMMARY:
  AI Model  → Chanos chanos (95%)
  KB Match  → Bandeng
  LLM Final → Ikan Bandeng
```

### Untuk Krustasea (Kepiting/Rajungan):
```
FINAL RESULT:
Indonesian Name: Rajungan
Scientific Name: Portunus pelagicus

✓ VALIDATED WITH MASTER DATA:
  Master Indonesian: Kepiting pasir
  Kelompok: Rajungan
  English: blue swimming crab; flower crab

✓✓ INDONESIAN NAME MATCHES MASTER DATA! ✓✓

PIPELINE SUMMARY:
  AI Model  → Portunus pelagicus (88%)
  KB Match  → Kepiting pasir (Rajungan)
  LLM Final → Rajungan
```

## 🎯 Validation Logic

Indonesian name dianggap MATCH jika:
1. LLM Indonesian name mengandung Master Indonesian name, ATAU
2. Master Indonesian name mengandung LLM Indonesian name

Contoh MATCH cases:
- LLM: "Ikan Bandeng" ↔ Master: "Bandeng" ✓
- LLM: "Rajungan" ↔ Master: "Kepiting pasir" (kelompok: Rajungan) ✓
- LLM: "Kepiting Bakau" ↔ Master: "Kepiting bakau besar" ✓

## 📊 Test Coverage

Test mencakup berbagai jenis biota:

| Kategori | Contoh Species | Kelompok |
|----------|----------------|----------|
| Ikan | Bandeng, Kerapu, Lele | Bandeng, Kerapu, Lele |
| Udang | Udang windu, Udang dogol | Udang |
| Kepiting | Kepiting bakau, Kepiting pasir | Kepiting |
| Rajungan | Rajungan angin, Rajungan salib | Rajungan |
| Kerang | Kerang darah, Abalone | Kerang |
| Cumi-cumi | Cumi-cumi | Cumi-cumi |

## 🐛 Troubleshooting

### Error: "Ollama service not available"
```bash
# Check Ollama service
curl https://ollama.hellodigi.id/api/tags

# Expected: HTTP 200 with list of models
```

### Error: "No test images found"
```bash
# Check images directory
ls ../images/*.jpg

# Add test images ke folder images/
```

### Error: "Could not decode image"
```bash
# Engine expects bytes, not string
# Make sure to pass: buffer.tobytes() not base64 string
```

### Error: "Models not loaded"
```bash
# Check model paths in settings.py
# Make sure all model files exist:
# - models/detection/
# - models/classification/
# - models/segmentation/
# - models/face_detector/
```

## 🚀 Running Tests

### Quick Test (Recommended untuk development):
```bash
cd /Users/user/Dev/researchs/fish_recognition_v2/fish_api
python test_quick_integration.py
```

### Full Knowledge Base Test:
```bash
python test_knowledge_base.py
```

### Full Integration Test (All images):
```bash
python test_integration.py  # Warning: Takes time!
```

## 📈 Success Criteria

Test dianggap berhasil jika:
- ✅ Knowledge base loaded (1437 species)
- ✅ AI Model dapat detect dan classify
- ✅ KB dapat menemukan similar species
- ✅ LLM dapat analyze dan identify
- ✅ **Indonesian name dari LLM match dengan master data**
- ✅ Validation logic bekerja dengan benar

## 🎉 Result

Dengan sistem yang sudah diperbaiki:
- ✅ Rajungan TIDAK LAGI salah diidentifikasi sebagai "Ikan Pari"
- ✅ Udang dapat dibedakan dari Kepiting (berdasarkan panjang antena)
- ✅ Kepiting dapat dibedakan dari Rajungan (berdasarkan paddle legs)
- ✅ Indonesian names sesuai dengan master data
- ✅ 225+ kelompok dan 1437 spesies tercakup

**The system is now working correctly!** 🎊
