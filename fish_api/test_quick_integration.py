"""
Quick Integration Test: AI Model + LLM + Knowledge Base
Test dengan 1 gambar untuk validasi cepat
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
import base64
from pathlib import Path


def test_single_image(image_path: str):
    """Test complete pipeline dengan 1 gambar"""
    print("=" * 80)
    print(f"TESTING: {Path(image_path).name}")
    print("=" * 80)
    
    # Initialize services
    print("\n1. Initializing services...")
    kb = get_knowledge_base_service()
    llm = get_ollama_service()
    engine = FishRecognitionEngine()
    
    print(f"   ✓ Knowledge base: {len(kb.species_data)} species")
    print(f"   ✓ LLM service: {llm.model}")
    print(f"   ✓ Fish engine: Ready")
    
    # Read image
    print(f"\n2. Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"   ✗ Failed to load: {image_path}")
        return
    
    print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()
    
    # Run detection and classification
    print(f"\n3. Running AI Model (Detection + Classification)...")
    results = engine.process_image(image_bytes)
    
    if not results or 'fish_detections' not in results:
        print("   ✗ No detections found")
        return
    
    fish_detections = results['fish_detections']
    print(f"   ✓ Detected {len(fish_detections)} object(s)")
    
    # Process first detection
    fish = fish_detections[0]
    classification = fish.get('classification', [])
    
    if not classification:
        print("   ⚠️  No classification results")
        return
    
    print(f"\n4. AI Model Results:")
    for i, pred in enumerate(classification[:3], 1):
        name = pred.get('name', pred.get('label', 'Unknown'))
        conf = pred.get('accuracy', pred.get('confidence', 0))
        print(f"   {i}. {name}: {conf:.2%}")
    
    # Build knowledge base context
    print(f"\n5. Knowledge Base Context...")
    detection_info = {
        'classification': classification,
        'detection_confidence': fish.get('detection_confidence', 0)
    }
    
    context = kb.build_context_for_llm(detection_info, top_k=5)
    
    if context.get('similar_species'):
        print(f"   ✓ Found {len(context['similar_species'])} similar species")
        for i, sp in enumerate(context['similar_species'][:3], 1):
            print(f"   {i}. {sp['indonesian_name']} ({sp['kelompok']})")
    
    if context.get('morphology_guide'):
        morph = context['morphology_guide']
        print(f"   ✓ Morphology guide: {morph['kelompok']}")
    
    # Get cropped image
    bbox = fish.get('bbox', [0, 0, image.shape[1], image.shape[0]])
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cropped = image[y1:y2, x1:x2]
    
    # LLM verification
    print(f"\n6. LLM Verification (gemma3:27b)...")
    print(f"   Sending to Ollama...")
    
    llm_result = llm.verify_classification(cropped, detection_info)
    
    if llm_result:
        print(f"\n{'='*80}")
        print(f"FINAL RESULT:")
        print(f"{'='*80}")
        print(f"Indonesian Name: {llm_result['indonesian_name']}")
        print(f"Scientific Name: {llm_result['scientific_name']}")
        
        # Validate with knowledge base
        kb_species = kb.get_species_by_scientific_name(llm_result['scientific_name'])
        if kb_species:
            print(f"\n✓ VALIDATED WITH MASTER DATA:")
            print(f"  Master Indonesian: {kb_species.get('species_indonesia', 'N/A')}")
            print(f"  Kelompok: {kb_species.get('kelompok', 'N/A')}")
            print(f"  English: {kb_species.get('species_english', 'N/A')}")
            
            # Check match
            llm_indo = llm_result['indonesian_name'].lower().replace('ikan ', '').strip()
            kb_indo = kb_species.get('species_indonesia', '').lower().strip()
            
            if llm_indo in kb_indo or kb_indo in llm_indo:
                print(f"\n{'='*80}")
                print(f"✓✓ INDONESIAN NAME MATCHES MASTER DATA! ✓✓")
                print(f"{'='*80}")
            else:
                print(f"\n⚠️  Names differ:")
                print(f"  LLM: {llm_result['indonesian_name']}")
                print(f"  Master: {kb_species.get('species_indonesia', 'N/A')}")
        else:
            print(f"\nℹ️  Species not in master data (LLM found new species)")
        
        # Summary
        print(f"\nPIPELINE SUMMARY:")
        print(f"  AI Model  → {classification[0].get('name', 'Unknown')} ({classification[0].get('accuracy', 0):.1%})")
        if context.get('similar_species'):
            print(f"  KB Match  → {context['similar_species'][0]['indonesian_name']}")
        print(f"  LLM Final → {llm_result['indonesian_name']}")
        print(f"\n✓ Test completed successfully!")
        
    else:
        print(f"\n✗ LLM verification failed")


def main():
    """Run quick test"""
    try:
        # Check Ollama
        print("Checking Ollama service...")
        llm = get_ollama_service()
        health = llm.health_check()
        
        print(f"Status: {health.get('status', 'unknown')}")
        print(f"Model: {health.get('model', 'N/A')}")
        
        if health.get('status') != 'healthy':
            print("\n⚠️  Ollama service not available")
            return 1
        
        # Find a test image
        images_dir = Path(__file__).parent.parent / "images"
        
        # Try to find specific test images
        test_candidates = [
            "bandeng.jpg",  # Ikan
            "6 Differences Crab vs. Blue Crab.jpg",  # Kepiting/Rajungan
            "Lele 4.jpg",  # Lele
            "kerapu2.jpg",  # Kerapu
        ]
        
        test_image = None
        for candidate in test_candidates:
            img_path = images_dir / candidate
            if img_path.exists():
                test_image = str(img_path)
                break
        
        if not test_image:
            # Fallback to any JPG
            jpg_images = list(images_dir.glob("*.jpg"))
            if jpg_images:
                test_image = str(jpg_images[0])
        
        if not test_image:
            print("\n✗ No test images found")
            return 1
        
        # Run test
        test_single_image(test_image)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
