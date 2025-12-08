"""
Test LLM dengan Gambar Asli
Test dengan gambar ikan dari folder images/
"""
import os
import sys
import django
import time
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fish_recognition_api.settings')
django.setup()

from recognition.services.ollama_llm_service import OllamaLLMService
import cv2
import numpy as np

# Initialize LLM service
print("="*80)
print("🧪 TEST LLM DENGAN GAMBAR ASLI")
print("="*80)
print()

llm_service = OllamaLLMService()

# Test images dari folder images/
test_images = [
    '../images/mujair.webp',
    '../images/bandeng.jpg',
    '../images/bandeng2.webp',
    '../images/kerapu.webp',
    '../images/kerapu2.jpg',
    '../images/buntal.webp',
    '../images/lempuk.jpg',
    '../images/Abudefduf saxatilis.webp'
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f'⚠️  File not found: {img_path}')
        continue
    
    print(f'📷 Testing: {os.path.basename(img_path)}')
    print('-' * 80)
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f'❌ Failed to load image')
        print()
        continue
    
    # Prepare detection info (simulate classification result - intentionally wrong for some)
    filename = os.path.basename(img_path).split('.')[0]
    
    # Simulate COMPLETELY WRONG predictions - both top predictions are wrong
    # Test if LLM can identify the fish WITHOUT relying on AI predictions
    if 'mujair' in filename.lower():
        # BOTH WRONG: Neither prediction is Mujair
        classification_results = [
            {'name': 'Clarias gariepinus', 'accuracy': 0.75, 'species_id': 500},  # Lele
            {'name': 'Pangasius hypophthalmus', 'accuracy': 0.20, 'species_id': 800}  # Patin
        ]
        print(f'🔬 Test case: Model prediksi Lele & Patin, seharusnya Mujair')
    elif 'bandeng' in filename.lower():
        # BOTH WRONG: Neither prediction is Bandeng
        classification_results = [
            {'name': 'Oreochromis niloticus', 'accuracy': 0.80, 'species_id': 451},  # Nila
            {'name': 'Lutjanus sp.', 'accuracy': 0.15, 'species_id': 600}  # Kakap
        ]
        print(f'🔬 Test case: Model prediksi Nila & Kakap, seharusnya Bandeng')
    elif 'kerapu' in filename.lower():
        # BOTH WRONG: Neither prediction is Kerapu
        classification_results = [
            {'name': 'Chanos chanos', 'accuracy': 0.70, 'species_id': 100},  # Bandeng
            {'name': 'Pampus argenteus', 'accuracy': 0.25, 'species_id': 700}  # Bawal
        ]
        print(f'🔬 Test case: Model prediksi Bandeng & Bawal, seharusnya Kerapu')
    elif 'buntal' in filename.lower():
        # BOTH WRONG: Neither prediction is Buntal
        classification_results = [
            {'name': 'Lutjanus sp.', 'accuracy': 0.65, 'species_id': 600},  # Kakap
            {'name': 'Epinephelus sp.', 'accuracy': 0.30, 'species_id': 200}  # Kerapu
        ]
        print(f'🔬 Test case: Model prediksi Kakap & Kerapu, seharusnya Buntal')
    elif 'lempuk' in filename.lower():
        # BOTH WRONG: Neither prediction is Lempuk/Djambal
        classification_results = [
            {'name': 'Chanos chanos', 'accuracy': 0.55, 'species_id': 100},  # Bandeng
            {'name': 'Oreochromis mossambicus', 'accuracy': 0.40, 'species_id': 450}  # Mujair
        ]
        print(f'🔬 Test case: Model prediksi Bandeng & Mujair, seharusnya Lempuk/Patin')
    else:
        # For unknown fish, give completely random wrong predictions
        classification_results = [
            {'name': 'Clarias gariepinus', 'accuracy': 0.60, 'species_id': 500},  # Lele
            {'name': 'Pangasius hypophthalmus', 'accuracy': 0.35, 'species_id': 800}  # Patin
        ]
        print(f'🔬 Test case: Model prediksi Lele & Patin (kemungkinan SALAH SEMUA)')
    
    print()
    
    detection_info = {
        'classification': classification_results,
        'detection_confidence': 0.95,
        'bbox': [0, 0, img.shape[1], img.shape[0]]
    }
    
    print(f'🤖 Model AI predicts: {classification_results[0]["name"]} ({classification_results[0]["accuracy"]:.1%})')
    
    # Call LLM
    start_time = time.time()
    result = llm_service.verify_classification(img, detection_info)
    duration = time.time() - start_time
    
    if result:
        print(f'✅ LLM Result:')
        print(f'   📚 Scientific: {result.get("scientific_name", "N/A")}')
        print(f'   🇮🇩 Indonesian: {result.get("indonesian_name", "N/A")}')
        print(f'   ⏱️  Time: {duration:.2f}s')
        
        # Check if LLM corrected the model
        model_name = classification_results[0]['name'].lower()
        llm_name = result.get('scientific_name', '').lower()
        if model_name not in llm_name and llm_name not in model_name:
            print(f'   🔄 LLM CORRECTED the model prediction!')
    else:
        print(f'❌ LLM failed or timeout')
    
    print()

print('=' * 80)
print('✅ Test completed!')
print('=' * 80)

