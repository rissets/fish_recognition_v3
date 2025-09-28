"""
Utility functions for image processing and validation
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def validate_image_format(image_data: bytes) -> bool:
    """
    Validate if the image data is in a supported format
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        True if format is supported, False otherwise
    """
    try:
        # Try to decode with OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False


def resize_image_maintain_aspect(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image as numpy array
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array in BGR format
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image from base64")
        
        return image
    except Exception as e:
        logger.error(f"Failed to convert base64 to image: {str(e)}")
        raise


def image_to_base64(image: np.ndarray, format: str = 'jpg', quality: int = 85) -> str:
    """
    Convert OpenCV image to base64 string
    
    Args:
        image: Image as numpy array in BGR format
        format: Output format ('jpg', 'png')
        quality: JPEG quality (1-100)
        
    Returns:
        Base64 encoded image string
    """
    try:
        if format.lower() == 'jpg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
        else:
            _, buffer = cv2.imencode('.png', image)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/{format};base64,{image_base64}"
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {str(e)}")
        raise


def draw_detection_results(image: np.ndarray, results: dict) -> np.ndarray:
    """
    Draw detection results on image for visualization
    
    Args:
        image: Input image in BGR format
        results: Recognition results from fish engine
        
    Returns:
        Image with drawn annotations
    """
    output_image = image.copy()
    
    # DEBUG: Log drawing process
    print(f"=== DRAW DETECTION RESULTS DEBUG ===")
    print(f"Input image shape: {image.shape}")
    print(f"Fish detections count: {len(results.get('fish_detections', []))}")
    
    # Draw fish detections
    for i, fish in enumerate(results.get('fish_detections', [])):
        print(f"Processing fish {i}: {fish}")
        bbox = fish['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        print(f"  BBox: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Draw segmentation polygons first (behind bounding box)
        segmentation = fish.get('segmentation', {})
        print(f"  Segmentation data: {segmentation}")
        
        if segmentation.get('has_segmentation', False) and segmentation.get('polygon_data'):
            print(f"  Drawing segmentation polygon...")
            print(f"  Polygon data: {segmentation['polygon_data']}")
            
            # Create semi-transparent overlay for segmentation
            overlay = output_image.copy()
            
            polygon_coords = segmentation['polygon_data']  # This should be a single polygon
            if len(polygon_coords) > 2:  # Need at least 3 points for a polygon
                # Convert to numpy array
                points = np.array(polygon_coords, dtype=np.int32)
                print(f"  Polygon points shape: {points.shape}")
                print(f"  Sample points: {points[:5]}")
                
                # Fill polygon with semi-transparent color
                cv2.fillPoly(overlay, [points], (0, 255, 255))  # Yellow fill
                
                # Draw polygon outline
                cv2.polylines(output_image, [points], True, (0, 200, 200), 2)  # Yellow outline
                print(f"  Polygon drawn successfully")
            else:
                print(f"  Not enough points for polygon: {len(polygon_coords)}")
            
            # Blend the overlay with original image for transparency
            alpha = 0.3  # Transparency factor
            output_image = cv2.addWeighted(output_image, 1 - alpha, overlay, alpha, 0)
        else:
            print(f"  No segmentation to draw")
        
        # Draw bounding box (on top of segmentation)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"  Bounding box drawn")
        
        # Prepare label
        classification = fish.get('classification', [])
        if classification:
            best_result = classification[0]
            label = f"{best_result['name']} ({best_result['accuracy']:.2%})"
        else:
            label = "Fish"
        
        # Add segmentation info to label if available
        if segmentation.get('has_segmentation', False):
            label += " [Segmented]"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(output_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Draw face detections
    for face in results.get('faces', []):
        bbox = face['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw face bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw face label
        cv2.putText(output_image, "Face", (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    print(f"=== END DRAW DETECTION RESULTS DEBUG ===")
    return output_image


def calculate_image_quality_score(image: np.ndarray) -> float:
    """
    Calculate a simple image quality score based on sharpness and contrast
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Quality score (0-1, higher is better)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast using standard deviation
        contrast = gray.std()
        
        # Normalize scores (these thresholds are empirical)
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        contrast_score = min(contrast / 50.0, 1.0)
        
        # Combined score
        quality_score = (sharpness_score + contrast_score) / 2.0
        
        return float(quality_score)
    except Exception as e:
        logger.warning(f"Failed to calculate image quality: {str(e)}")
        return 0.5  # Default score


def is_image_too_dark(image: np.ndarray, threshold: float = 0.2) -> bool:
    """
    Check if image is too dark for good recognition
    
    Args:
        image: Input image in BGR format
        threshold: Darkness threshold (0-1)
        
    Returns:
        True if image is too dark
    """
    try:
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean() / 255.0
        
        return mean_brightness < threshold
    except Exception:
        return False


def is_image_too_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if image is too blurry for good recognition
    
    Args:
        image: Input image in BGR format
        threshold: Blur threshold (lower values = more blurry)
        
    Returns:
        True if image is too blurry
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (measure of blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var < threshold
    except Exception:
        return False


class ImageQualityValidator:
    """Class for comprehensive image quality validation"""
    
    def __init__(self, 
                 min_resolution: Tuple[int, int] = (224, 224),
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 min_quality_score: float = 0.3,
                 max_darkness_threshold: float = 0.2,
                 min_sharpness_threshold: float = 100.0):
        self.min_resolution = min_resolution
        self.max_file_size = max_file_size
        self.min_quality_score = min_quality_score
        self.max_darkness_threshold = max_darkness_threshold
        self.min_sharpness_threshold = min_sharpness_threshold
    
    def validate(self, image_data: bytes) -> dict:
        """
        Comprehensive image validation
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Validation results with recommendations
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "quality_score": 0.0
        }
        
        try:
            # Check file size
            if len(image_data) > self.max_file_size:
                results["errors"].append(f"File too large: {len(image_data)} bytes > {self.max_file_size} bytes")
                results["valid"] = False
            
            # Validate image format
            if not validate_image_format(image_data):
                results["errors"].append("Invalid or unsupported image format")
                results["valid"] = False
                return results
            
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Check resolution
            height, width = image.shape[:2]
            if height < self.min_resolution[1] or width < self.min_resolution[0]:
                results["errors"].append(f"Resolution too low: {width}x{height} < {self.min_resolution[0]}x{self.min_resolution[1]}")
                results["valid"] = False
            
            # Calculate quality score
            quality_score = calculate_image_quality_score(image)
            results["quality_score"] = quality_score
            
            if quality_score < self.min_quality_score:
                results["warnings"].append(f"Low image quality: {quality_score:.2f} < {self.min_quality_score}")
                results["recommendations"].append("Try taking a clearer, higher contrast image")
            
            # Check if too dark
            if is_image_too_dark(image, self.max_darkness_threshold):
                results["warnings"].append("Image appears to be too dark")
                results["recommendations"].append("Increase lighting or brightness")
            
            # Check if too blurry
            if is_image_too_blurry(image, self.min_sharpness_threshold):
                results["warnings"].append("Image appears to be blurry")
                results["recommendations"].append("Take a sharper, more focused image")
            
        except Exception as e:
            results["errors"].append(f"Validation failed: {str(e)}")
            results["valid"] = False
        
        return results