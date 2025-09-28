#!/usr/bin/env python3
"""
Test script for Fish Recognition API
"""

import requests
import base64
import json
from pathlib import Path

def test_recognize_with_file(image_path: str):
    """Test recognition with file upload"""
    url = "http://localhost:8001/api/v1/recognize/"

    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'include_faces': 'true',
            'include_segmentation': 'true',
            'include_visualization': 'false'
        }

        print(f"Testing file upload with {image_path}")
        response = requests.post(url, files=files, data=data)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Found {len(result.get('fish_detections', []))} fish detections")
            for i, fish in enumerate(result.get('fish_detections', [])):
                print(f"  Fish {i+1}: {fish.get('classification', [{}])[0].get('name', 'Unknown')}")
        else:
            print(f"Error: {response.text}")

def test_recognize_with_base64(image_path: str):
    """Test recognition with base64 encoding"""
    url = "http://localhost:8001/api/v1/recognize/"

    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Convert to base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    data = {
        'image_base64': f"data:image/jpeg;base64,{image_base64}",
        'include_faces': True,
        'include_segmentation': True,
        'include_visualization': False
    }

    print(f"Testing base64 upload with {image_path}")
    response = requests.post(url, json=data)

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Found {len(result.get('fish_detections', []))} fish detections")
        for i, fish in enumerate(result.get('fish_detections', [])):
            print(f"  Fish {i+1}: {fish.get('classification', [{}])[0].get('name', 'Unknown')}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    # Test with available images
    test_images = [
        "../backups/images/mujair.jpeg",
        "../backups/images/Ikan Mujair.jpg"
    ]

    for image_path in test_images:
        if Path(image_path).exists():
            print("\n" + "="*50)
            test_recognize_with_file(image_path)
            print("\n" + "-"*30)
            test_recognize_with_base64(image_path)
            break  # Test just one image
        else:
            print(f"Image not found: {image_path}")