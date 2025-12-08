#!/usr/bin/env python3
"""
Test LLM as Primary Output
Verify that LLM result appears as top prediction in all endpoints
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fish_recognition_api.settings')
django.setup()

from recognition.ml_models.fish_engine import get_fish_engine
import cv2

def test_llm_primary_output():
    """Test that LLM result is primary output"""
    
    print("=" * 80)
    print("🧪 TEST LLM AS PRIMARY OUTPUT")
    print("=" * 80)
    
    # Load test image
    test_image = "/Users/user/Dev/researchs/fish_recognition_v2/images/bandeng.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"\n📷 Loading test image: {test_image}")
    
    # Read image
    image_bgr = cv2.imread(test_image)
    if image_bgr is None:
        print("❌ Failed to load image")
        return
    
    # Encode to bytes
    success, buffer = cv2.imencode('.jpg', image_bgr)
    if not success:
        print("❌ Failed to encode image")
        return
    
    image_data = buffer.tobytes()
    print(f"✅ Image loaded: {image_bgr.shape}, {len(image_data)} bytes")
    
    # Get engine and process
    print("\n🔧 Processing with Fish Engine...")
    engine = get_fish_engine()
    
    results = engine.process_image(
        image_data=image_data,
        include_faces=True,
        include_segmentation=False
    )
    
    print(f"\n📊 Results:")
    print(f"   Fish detected: {len(results.get('fish_detections', []))}")
    print(f"   Processing time: {results.get('total_processing_time', 0):.2f}s")
    
    # Check each fish detection
    for i, fish in enumerate(results.get('fish_detections', [])):
        print(f"\n🐟 Fish #{i+1}:")
        print(f"   Confidence: {fish.get('confidence', 0):.2%}")
        
        # Check classification
        classifications = fish.get('classification', [])
        print(f"\n   📋 Classifications ({len(classifications)} total):")
        
        for j, pred in enumerate(classifications[:3]):  # Show top 3
            source = pred.get('source', 'unknown')
            name = pred.get('name', 'Unknown')
            accuracy = pred.get('accuracy', 0)
            
            indicator = "🎯" if j == 0 else "  "
            source_label = "LLM" if source == "llm" else "MODEL"
            
            print(f"   {indicator} [{j+1}] {name}")
            print(f"        Source: {source_label} | Accuracy: {accuracy:.2%}")
            
            if source == "llm":
                scientific_name = pred.get('scientific_name', 'Unknown')
                print(f"        Scientific: {scientific_name}")
        
        # Check LLM verification
        llm_verification = fish.get('llm_verification')
        if llm_verification:
            print(f"\n   🤖 LLM Verification:")
            print(f"      Scientific: {llm_verification.get('scientific_name', 'N/A')}")
            print(f"      Indonesian: {llm_verification.get('indonesian_name', 'N/A')}")
            print(f"      Processing Time: {llm_verification.get('processing_time', 0):.2f}s")
        else:
            print(f"\n   ⚠️  No LLM verification")
        
        # Verify primary output
        print(f"\n   ✅ Verification:")
        if classifications:
            primary = classifications[0]
            if primary.get('source') == 'llm':
                print(f"      ✓ LLM is PRIMARY output")
                print(f"      ✓ Display name: {primary.get('name')}")
                print(f"      ✓ Scientific name: {primary.get('scientific_name')}")
            else:
                print(f"      ✗ Model is primary (LLM should be first!)")
        else:
            print(f"      ✗ No classifications found")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_llm_primary_output()
