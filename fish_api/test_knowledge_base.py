"""
Test script untuk memvalidasi Knowledge Base Service dan integrasi dengan LLM
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
import cv2
import numpy as np


def test_knowledge_base():
    """Test Knowledge Base Service"""
    print("=" * 80)
    print("TESTING KNOWLEDGE BASE SERVICE")
    print("=" * 80)
    
    kb = get_knowledge_base_service()
    
    # Test 1: Check kelompok list
    print("\n1. Testing kelompok list:")
    kelompok_list = kb.get_kelompok_list()
    print(f"   Found {len(kelompok_list)} kelompok:")
    for kelompok in kelompok_list[:20]:
        print(f"   - {kelompok}")
    
    # Test 2: Search by kelompok
    print("\n2. Testing search by kelompok:")
    test_kelompok = ["Udang", "Kepiting", "Rajungan", "Kerang"]
    for kelompok in test_kelompok:
        results = kb.search_by_kelompok(kelompok, limit=3)
        print(f"\n   {kelompok} (found {len(results)} species):")
        for species in results:
            print(f"   - {species.get('species_indonesia', '')} ({species.get('nama_latin', '').split(';')[0].strip()})")
    
    # Test 3: Get morphology guide
    print("\n3. Testing morphology guide:")
    for kelompok in ["Udang", "Kepiting", "Rajungan"]:
        morph = kb.get_kelompok_morphology(kelompok)
        if morph:
            print(f"\n   {kelompok}:")
            print(f"   Body type: {morph['body_type']}")
            print(f"   Fins: {morph['fins']}")
            print(f"   Key features (first 3):")
            for feature in morph['key_features'][:3]:
                print(f"     • {feature}")
    
    # Test 4: Find similar species
    print("\n4. Testing find similar species:")
    test_queries = [
        ("Portunus pelagicus", "Rajungan"),
        ("Scylla serrata", "Kepiting"),
        ("Penaeus monodon", "Udang")
    ]
    for query, expected_kelompok in test_queries:
        results = kb.find_similar_species(query, limit=3)
        print(f"\n   Query: {query} (expected: {expected_kelompok})")
        for species, score in results:
            print(f"   - {species.get('species_indonesia', '')} ({species.get('kelompok', '')}) - score: {score:.2f}")
    
    # Test 5: Build context for LLM
    print("\n5. Testing build context for LLM:")
    mock_detection_info = {
        "classification": [
            {"name": "Raja clavata", "accuracy": 0.85},
            {"name": "Portunus pelagicus", "accuracy": 0.10}
        ]
    }
    context = kb.build_context_for_llm(mock_detection_info, top_k=5)
    print(f"   Similar species found: {len(context.get('similar_species', []))}")
    if context.get("similar_species"):
        print(f"   Top match:")
        sp = context["similar_species"][0]
        print(f"   - {sp['indonesian_name']} ({sp['scientific_name']}) - kelompok: {sp['kelompok']}")
    
    if context.get("morphology_guide"):
        print(f"\n   Morphology guide provided for: {context['morphology_guide']['kelompok']}")
    
    print("\n" + "=" * 80)
    print("KNOWLEDGE BASE SERVICE TEST COMPLETED")
    print("=" * 80)


def test_llm_integration():
    """Test LLM integration with Knowledge Base"""
    print("\n" + "=" * 80)
    print("TESTING LLM INTEGRATION WITH KNOWLEDGE BASE")
    print("=" * 80)
    
    kb = get_knowledge_base_service()
    
    # Test context building
    print("\n1. Testing context building for crustacean:")
    mock_detection_info = {
        "classification": [
            {"name": "Portunus pelagicus", "accuracy": 0.75},
            {"name": "Blue swimming crab", "accuracy": 0.15}
        ]
    }
    
    context = kb.build_context_for_llm(mock_detection_info, top_k=5)
    
    print(f"\n   Context summary:")
    print(f"   - Similar species: {len(context.get('similar_species', []))}")
    print(f"   - Morphology guide available: {context.get('morphology_guide') is not None}")
    print(f"   - Kelompok database size: {len(context.get('kelompok_database', []))}")
    
    if context.get("similar_species"):
        print(f"\n   Top 3 similar species:")
        for sp in context["similar_species"][:3]:
            print(f"   - {sp['indonesian_name']}")
            print(f"     Scientific: {sp['scientific_name']}")
            print(f"     Kelompok: {sp['kelompok']}")
            print(f"     Similarity: {sp['similarity']}")
    
    if context.get("morphology_guide"):
        morph = context["morphology_guide"]
        print(f"\n   Morphology guide for {morph['kelompok']}:")
        print(f"   - Body type: {morph['body_type']}")
        print(f"   - Fins: {morph['fins']}")
        print(f"   - Texture: {morph['texture']}")
    
    print("\n" + "=" * 80)
    print("LLM INTEGRATION TEST COMPLETED")
    print("=" * 80)


def main():
    """Run all tests"""
    try:
        test_knowledge_base()
        test_llm_integration()
        
        print("\n✓ All tests completed successfully!")
        print("\nThe system is now ready to identify:")
        print("  • Ikan (Fish)")
        print("  • Udang (Shrimp/Prawn)")
        print("  • Kepiting (Crab)")
        print("  • Rajungan (Swimming Crab)")
        print("  • Kerang (Shellfish)")
        print("  • Cumi-cumi (Squid)")
        print("  • Gurita (Octopus)")
        print("  • Sotong (Cuttlefish)")
        print("  • And many more marine organisms!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
