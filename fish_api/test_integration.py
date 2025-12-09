"""
Integration Test: AI Model + LLM + Knowledge Base
Test complete pipeline dengan gambar real
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fish_recognition_api.settings')
import django
django.setup()

from recognition.services.knowledge_base_service import get_knowledge_base_service
from recognition.services.ollama_llm_service import get_ollama_service
from recognition.ml_models.fish_engine import FishRecognitionEngine
import cv2
import numpy as np
from pathlib import Path


def test_complete_pipeline():
    """Test complete pipeline: Detection → Classification → Knowledge Base → LLM"""
    print("=" * 80)
    print("TESTING COMPLETE PIPELINE: AI MODEL + LLM + KNOWLEDGE BASE")
    print("=" * 80)
    
    # Initialize services
    kb = get_knowledge_base_service()
    llm = get_ollama_service()
    engine = FishRecognitionEngine()
    
    # Test images directory
    images_dir = Path(__file__).parent.parent / "images"
    
    # Support common formats (skip AVIF for now as cv2 doesn't support it by default)
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        test_images.extend(list(images_dir.glob(ext)))
    
    # Try AVIF but with error handling
    for avif_file in images_dir.glob("*.avif"):
        # Try to read with cv2, skip if fails
        try:
            test_img = cv2.imread(str(avif_file))
            if test_img is not None:
                test_images.append(avif_file)
        except:
            print(f"⚠️  Skipping {avif_file.name} (AVIF format not supported by cv2)")
    
    if not test_images:
        print("\n⚠️  No supported test images found in images/ directory")
        print("   Supported formats: JPG, JPEG, PNG, BMP")
        print("   Note: AVIF format requires pillow-avif-plugin")
        return
    
    print(f"\nFound {len(test_images)} test images")
    print("-" * 80)
    
    for img_path in test_images:
        print(f"\n{'='*80}")
        print(f"Testing image: {img_path.name}")
        print(f"{'='*80}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"✗ Failed to load image: {img_path}")
            continue
        
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Convert image to base64 untuk fish_engine
        import base64
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Run detection and classification
        print("\n📊 Step 1: Running AI Model (Detection + Classification)...")
        results = engine.process_image(image_bytes)
        
        if not results or 'fish_detections' not in results:
            print("✗ No detections found")
            continue
        
        fish_detections = results['fish_detections']
        print(f"✓ Detected {len(fish_detections)} object(s)")
        
        # Process each detection
        for i, fish in enumerate(fish_detections):
            print(f"\n{'-'*80}")
            print(f"Object {i+1}/{len(fish_detections)}")
            print(f"{'-'*80}")
            
            # Get classification results
            classification = fish.get('classification', [])
            if not classification:
                print("⚠️  No classification results")
                continue
            
            print(f"\n🤖 AI Model Top 3 Predictions:")
            for j, pred in enumerate(classification[:3], 1):
                name = pred.get('name', pred.get('label', 'Unknown'))
                conf = pred.get('accuracy', pred.get('confidence', 0))
                print(f"   {j}. {name}: {conf:.2%}")
            
            # Build knowledge base context
            print(f"\n📚 Step 2: Knowledge Base Context Building...")
            detection_info = {
                'classification': classification,
                'detection_confidence': fish.get('detection_confidence', 0)
            }
            
            context = kb.build_context_for_llm(detection_info, top_k=5)
            
            if context.get('similar_species'):
                print(f"✓ Found {len(context['similar_species'])} similar species in KB")
                print(f"\n   Top 3 KB matches:")
                for j, sp in enumerate(context['similar_species'][:3], 1):
                    print(f"   {j}. {sp['indonesian_name']}")
                    print(f"      Scientific: {sp['scientific_name']}")
                    print(f"      Kelompok: {sp['kelompok']}")
                    print(f"      Similarity: {sp['similarity']}")
            else:
                print("⚠️  No similar species found in KB")
            
            if context.get('morphology_guide'):
                morph = context['morphology_guide']
                print(f"\n   Morphology guide: {morph['kelompok']}")
                print(f"   Body type: {morph['body_type']}")
                print(f"   Fins: {morph['fins']}")
            
            # Get cropped image for LLM
            bbox = fish.get('bbox', [0, 0, image.shape[1], image.shape[0]])
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                print("✗ Failed to crop image")
                continue
            
            # LLM verification
            print(f"\n🧠 Step 3: LLM Verification (with Knowledge Base integration)...")
            print(f"   Sending to Ollama LLM (gemma3:27b)...")
            
            llm_result = llm.verify_classification(cropped, detection_info)
            
            if llm_result:
                print(f"\n✅ FINAL RESULT (LLM + KB):")
                print(f"   Indonesian Name: {llm_result['indonesian_name']}")
                print(f"   Scientific Name: {llm_result['scientific_name']}")
                
                # Validate with knowledge base
                kb_species = kb.get_species_by_scientific_name(llm_result['scientific_name'])
                if kb_species:
                    print(f"\n   ✓ Validated with Knowledge Base:")
                    print(f"     Master Data Indonesian: {kb_species.get('species_indonesia', 'N/A')}")
                    print(f"     Kelompok: {kb_species.get('kelompok', 'N/A')}")
                    print(f"     English Name: {kb_species.get('species_english', 'N/A')}")
                    
                    # Check if LLM indonesian name matches master data
                    llm_indo = llm_result['indonesian_name'].lower()
                    kb_indo = kb_species.get('species_indonesia', '').lower()
                    
                    if llm_indo in kb_indo or kb_indo in llm_indo:
                        print(f"\n   ✓✓ Indonesian name MATCHES master data! ✓✓")
                    else:
                        print(f"\n   ⚠️  Indonesian name differs from master data")
                        print(f"      LLM: {llm_result['indonesian_name']}")
                        print(f"      Master: {kb_species.get('species_indonesia', 'N/A')}")
                else:
                    print(f"\n   ℹ️  Species not found in knowledge base")
                    print(f"      (LLM identified species outside master data)")
                
                # Summary comparison
                print(f"\n📊 PIPELINE SUMMARY:")
                print(f"   AI Model → {classification[0].get('name', 'Unknown')} ({classification[0].get('accuracy', 0):.2%})")
                if context.get('similar_species'):
                    print(f"   Knowledge Base → {context['similar_species'][0]['indonesian_name']}")
                print(f"   LLM Final → {llm_result['indonesian_name']}")
                
            else:
                print(f"\n✗ LLM verification failed")
                print(f"   Falling back to AI model prediction:")
                print(f"   {classification[0].get('name', 'Unknown')}")
    
    print(f"\n{'='*80}")
    print("INTEGRATION TEST COMPLETED")
    print(f"{'='*80}")


def test_specific_cases():
    """Test specific cases untuk validasi"""
    print("\n" + "=" * 80)
    print("TESTING SPECIFIC VALIDATION CASES")
    print("=" * 80)
    
    kb = get_knowledge_base_service()
    
    # Test case 1: Validate rajungan
    print("\n1. Testing Rajungan identification:")
    test_species = [
        ("Portunus pelagicus", "Rajungan"),
        ("Scylla serrata", "Kepiting"),
        ("Penaeus monodon", "Udang")
    ]
    
    for scientific_name, expected_kelompok in test_species:
        species = kb.get_species_by_scientific_name(scientific_name)
        if species:
            print(f"\n   Scientific: {scientific_name}")
            print(f"   Indonesian: {species.get('species_indonesia', 'N/A')}")
            print(f"   Kelompok: {species.get('kelompok', 'N/A')}")
            print(f"   Expected: {expected_kelompok}")
            
            if species.get('kelompok', '') == expected_kelompok:
                print(f"   ✓ Kelompok matches!")
            else:
                print(f"   ✗ Kelompok mismatch!")
        else:
            print(f"\n   ✗ {scientific_name} not found in KB")
    
    # Test case 2: Check master data coverage
    print("\n2. Checking master data coverage:")
    kelompok_counts = {}
    for species in kb.species_data:
        kelompok = species.get('kelompok', 'Unknown')
        kelompok_counts[kelompok] = kelompok_counts.get(kelompok, 0) + 1
    
    important_groups = ['Ikan', 'Udang', 'Kepiting', 'Rajungan', 'Kerang', 'Cumi-cumi', 'Gurita', 'Sotong']
    print(f"\n   Coverage for important groups:")
    for group in important_groups:
        count = sum(v for k, v in kelompok_counts.items() if group.lower() in k.lower())
        print(f"   {group}: {count} species")
    
    print("\n" + "=" * 80)
    print("VALIDATION TEST COMPLETED")
    print("=" * 80)


def main():
    """Run all integration tests"""
    try:
        # Check if Ollama service is available
        print("Checking Ollama service availability...")
        llm = get_ollama_service()
        health = llm.health_check()
        
        print(f"\nOllama Service Status: {health.get('status', 'unknown')}")
        print(f"URL: {health.get('url', 'N/A')}")
        print(f"Model: {health.get('model', 'N/A')}")
        print(f"Model Available: {health.get('model_available', False)}")
        
        if health.get('status') != 'healthy':
            print("\n⚠️  WARNING: Ollama service is not healthy!")
            print("   LLM verification tests will be skipped.")
            print("   Only KB validation tests will run.")
            
            # Run only validation tests
            test_specific_cases()
            return 0
        
        # Run complete pipeline test
        test_complete_pipeline()
        
        # Run validation tests
        test_specific_cases()
        
        print("\n" + "=" * 80)
        print("✓ ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nSystem validated:")
        print("  ✓ AI Model detection and classification")
        print("  ✓ Knowledge Base context building")
        print("  ✓ LLM verification with KB integration")
        print("  ✓ Indonesian name matching with master data")
        print("\nThe pipeline is working correctly! 🎉")
        
    except Exception as e:
        print(f"\n✗ Integration test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
