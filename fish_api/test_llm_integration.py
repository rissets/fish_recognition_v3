"""
Test Script untuk LLM Integration
Tests Ollama gamma3 integration untuk enhanced fish classification
"""

import os
import sys
import requests
import json
import base64
from pathlib import Path

# API Base URL
API_BASE_URL = "http://localhost:8001/api/recognition"


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_result(success, message):
    """Print formatted result"""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{status}: {message}")


def test_llm_health():
    """Test 1: Check LLM health and configuration"""
    print_header("Test 1: LLM Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}/config/llm/")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"LLM Enabled: {data.get('enabled')}")
        print(f"Service Available: {data.get('service_available')}")
        
        if 'health' in data and data['health']:
            health = data['health']
            print(f"Health Status: {health.get('status')}")
            print(f"Model: {health.get('model')}")
            print(f"URL: {health.get('url')}")
            print(f"Model Available: {health.get('model_available')}")
        
        success = response.status_code == 200 and data.get('service_available')
        print_result(success, "LLM service is configured and available" if success else "LLM service not available")
        
        return success
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        return False


def test_health_endpoint():
    """Test 2: Check main health endpoint includes LLM info"""
    print_header("Test 2: Main Health Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health/")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"API Status: {data.get('status')}")
        print(f"Models Loaded: {data.get('models_loaded')}")
        
        if 'llm_enhancement' in data:
            llm_info = data['llm_enhancement']
            print(f"LLM Enabled: {llm_info.get('enabled')}")
            print(f"LLM Service Available: {llm_info.get('service_available')}")
        
        success = response.status_code == 200
        print_result(success, "Health endpoint accessible")
        
        return success
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        return False


def test_llm_toggle():
    """Test 3: Toggle LLM configuration"""
    print_header("Test 3: LLM Configuration Toggle")
    
    try:
        # Disable LLM
        print("\nDisabling LLM...")
        response = requests.post(
            f"{API_BASE_URL}/config/llm/",
            json={"enabled": False}
        )
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        disabled = response.status_code == 200 and not data['config']['enabled']
        print_result(disabled, "LLM disabled successfully" if disabled else "Failed to disable LLM")
        
        # Enable LLM
        print("\nEnabling LLM...")
        response = requests.post(
            f"{API_BASE_URL}/config/llm/",
            json={"enabled": True}
        )
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        enabled = response.status_code == 200 and data['config']['enabled']
        print_result(enabled, "LLM enabled successfully" if enabled else "Failed to enable LLM")
        
        return disabled and enabled
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        return False


def test_recognition_with_llm(image_path=None):
    """Test 4: Fish recognition with LLM verification"""
    print_header("Test 4: Recognition with LLM Verification")
    
    # Try to find a test image
    if image_path is None:
        # Look for test images in common locations
        possible_paths = [
            Path("media/test_fish.jpg"),
            Path("../backups/dataset/test_image.jpg"),
            Path("test_image.jpg")
        ]
        
        for path in possible_paths:
            if path.exists():
                image_path = str(path)
                break
    
    if image_path is None or not Path(image_path).exists():
        print_result(False, "No test image found. Please provide image path.")
        print("Usage: test_recognition_with_llm('/path/to/fish/image.jpg')")
        return False
    
    try:
        print(f"Using image: {image_path}")
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Send recognition request
        files = {'image': ('test.jpg', image_data, 'image/jpeg')}
        data = {
            'include_faces': 'true',
            'include_segmentation': 'true',
            'include_visualization': 'false'
        }
        
        print("Sending recognition request...")
        response = requests.post(
            f"{API_BASE_URL}/recognize/",
            files=files,
            data=data,
            timeout=60  # Longer timeout for LLM processing
        )
        
        result = response.json()
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Success: {result.get('success')}")
        print(f"Fish Detected: {len(result.get('fish_detections', []))}")
        
        if result.get('fish_detections'):
            for i, detection in enumerate(result['fish_detections']):
                print(f"\n--- Fish {i+1} ---")
                print(f"Detection Confidence: {detection.get('confidence', 0):.2%}")
                
                # Classification results
                if detection.get('classification'):
                    top_class = detection['classification'][0]
                    print(f"Top Classification: {top_class['label']} ({top_class['confidence']:.2%})")
                
                # LLM verification
                if detection.get('llm_verification'):
                    llm_result = detection['llm_verification']
                    if 'error' in llm_result:
                        print(f"LLM Verification: Error - {llm_result['error']}")
                    else:
                        print(f"LLM Scientific Name: {llm_result.get('scientific_name')}")
                        print(f"LLM Indonesian Name: {llm_result.get('indonesian_name')}")
                        print(f"LLM Processing Time: {llm_result.get('processing_time', 0):.2f}s")
                else:
                    print("LLM Verification: Not available")
        
        success = response.status_code == 200 and result.get('success')
        print_result(success, "Recognition completed successfully" if success else "Recognition failed")
        
        return success
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        return False


def test_performance_stats():
    """Test 5: Check performance statistics include LLM"""
    print_header("Test 5: Performance Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats/")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print("\nAvailable Statistics:")
        for operation, stats in data.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"  {operation}:")
                print(f"    - Mean: {stats['mean']:.3f}s")
                print(f"    - Count: {stats['count']}")
                print(f"    - Min: {stats['min']:.3f}s")
                print(f"    - Max: {stats['max']:.3f}s")
        
        success = response.status_code == 200
        print_result(success, "Performance stats accessible")
        
        return success
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        return False


def run_all_tests(image_path=None):
    """Run all tests"""
    print("\n" + "="*60)
    print("  LLM Integration Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("LLM Health Check", test_llm_health()))
    results.append(("Main Health Endpoint", test_health_endpoint()))
    results.append(("LLM Configuration Toggle", test_llm_toggle()))
    results.append(("Performance Stats", test_performance_stats()))
    
    # Recognition test is optional (requires image)
    if image_path:
        results.append(("Recognition with LLM", test_recognition_with_llm(image_path)))
    
    # Print summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM Integration")
    parser.add_argument('--image', type=str, help='Path to test image for recognition test')
    parser.add_argument('--url', type=str, default='http://localhost:8001', help='API base URL')
    parser.add_argument('--test', type=str, choices=['health', 'config', 'recognition', 'stats', 'all'],
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    # Update global API URL
    global API_BASE_URL
    API_BASE_URL = f"{args.url}/api/recognition"
    
    # Run specific test or all tests
    if args.test == 'health':
        test_llm_health()
    elif args.test == 'config':
        test_llm_toggle()
    elif args.test == 'recognition':
        test_recognition_with_llm(args.image)
    elif args.test == 'stats':
        test_performance_stats()
    else:
        run_all_tests(args.image)


if __name__ == "__main__":
    main()
