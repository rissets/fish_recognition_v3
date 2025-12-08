"""
Optimized Fish Recognition Engine
Combines detection, classification, and segmentation for accurate fish recognition
"""

import os
import sys
import cv2
import numpy as np
import torch
import logging
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from collections import defaultdict
import threading
from django.conf import settings
from pathlib import Path


def _ensure_project_root_on_path():
    current_dir = Path(__file__).resolve().parent
    for candidate in [current_dir] + list(current_dir.parents):
        if (candidate / 'models').is_dir() and (candidate / 'fish_api').is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
            return


_ensure_project_root_on_path()

# Import original model classes
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference, YOLOResult
from models.segmentation.inference import Inference
from models.face_detector.inference import YOLOInference as FaceInference

from ..utils.image_utils import draw_detection_results, image_to_base64
from ..services.ollama_llm_service import get_ollama_service

logger = logging.getLogger(__name__)


class FishRecognitionEngine:
    """
    Optimized Fish Recognition Engine that combines detection, classification, and segmentation
    with caching and performance optimizations for real-time processing.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of the engine exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.device = settings.FISH_MODEL_SETTINGS.get('DEVICE', 'cpu')
        self.models_loaded = False
        self.model_cache = {}
        
        # Face filtering configuration
        self.face_filter_enabled = True
        self.face_filter_iou_threshold = 0.3  # IoU threshold for face-fish overlap
        
        # LLM Enhancement configuration
        self.llm_enabled = True
        self.llm_service = None
        
        # Performance tracking
        self.processing_stats = defaultdict(list)
        
        logger.info("Initializing Fish Recognition Engine...")
        self._load_models()
        self._initialize_llm()
    
    def _load_models(self):
        """Load all ML models with error handling"""
        try:
            model_settings = settings.FISH_MODEL_SETTINGS
            
            # Load Classification Model
            logger.info("Loading classification model...")
            classifier_config = {
                "dataset": {
                    "path": model_settings['CLASSIFICATION_DATABASE']
                },
                "model": {
                    "path": model_settings['CLASSIFICATION_MODEL'],
                    "device": self.device
                },
                "log_level": "INFO"
            }
            self.classifier = EmbeddingClassifier(classifier_config)
            
            # Load Detection Model
            logger.info("Loading detection model...")
            self.detector = YOLOInference(
                model_path=model_settings['DETECTION_MODEL'],
                imsz=(640, 640),
                conf_threshold=model_settings.get('CONFIDENCE_THRESHOLD', 0.5),
                nms_threshold=model_settings.get('NMS_THRESHOLD', 0.3),
                yolo_ver='v8'
            )
            
            # Force yolo_ver to v8 and monkey-patch predict to ensure it uses v8
            self.detector.yolo_ver = 'v8'
            
            def custom_predict(im_bgr):
                if isinstance(im_bgr, np.ndarray):
                    im_bgr = [im_bgr]
                    
                input_imgs, params = self.detector.preprocess(im_bgr)
                
                with torch.no_grad():
                    predictions = self.detector.model(input_imgs)
                
                final_pred = []
                for bbox_id in range(len(predictions)):
                    # Always use v8 postprocessing
                    filtered_boxes = self.detector.v8postprocess(predictions[bbox_id])
                    
                    if len(filtered_boxes) == 0:
                        final_pred.append([])
                    else:
                        boxes = self.detector.scale_coords_back(im_bgr[bbox_id].shape[:2], filtered_boxes, params[bbox_id])
                        final_pred.append([YOLOResult(box, im_bgr[bbox_id]) for box in boxes])
                return final_pred
            
            # Replace predict method
            self.detector.predict = custom_predict
            
            # Monkey patch the v10postprocess to handle the actual tensor format
            def custom_v10postprocess(self, predictions):
                # If predictions has shape (..., 8400), assume it's YOLOv8 format flattened
                if predictions.shape[-1] == 8400:
                    # Reshape to (5, 1680) assuming 1680 predictions
                    predictions = predictions.view(5, -1)
                    # Use v8 postprocessing logic
                    x_center, y_center, width, height, confidence = predictions
                else:
                    # Original v10 logic
                    boxes, scores, labels = predictions.split([4, 1, 1], dim=-1)
                    return self.v10postprocess_original(predictions)
                
                # v8 postprocessing logic
                boxes = np.stack((x_center, y_center, width, height, confidence), axis=1)
                boxes = boxes[boxes[:, 4] > self.conf_threshold]
                
                if len(boxes) > 0:
                    selected_boxes = boxes[:, :4]
                    selected_scores = boxes[:, 4:5]
                    boxes_scores = np.hstack([selected_boxes, selected_scores])
                    indices = self.nms(boxes_scores)
                    return boxes_scores[indices]
                return np.array([])
            
            # Replace the v10postprocess with custom one
            self.detector.v10postprocess_original = self.detector.v10postprocess
            self.detector.v10postprocess = lambda predictions: custom_v10postprocess(self.detector, predictions)
            # Ensure yolo_ver is set correctly
            self.detector.yolo_ver = 'v8'
            
            # Load Segmentation Model
            logger.info("Loading segmentation model...")
            self.segmentator = Inference(
                model_path=model_settings['SEGMENTATION_MODEL'],
                image_size=416,
                threshold=model_settings.get('SEGMENTATION_THRESHOLD', 0.5)
            )
            
            # Load Face Detection Model
            logger.info("Loading face detection model...")
            self.face_detector = FaceInference(
                model_path=model_settings['FACE_DETECTION_MODEL'],
                imsz=(640, 640),
                conf_threshold=0.69,
                nms_threshold=0.5,
                yolo_ver='v8'
            )
            
            self.models_loaded = True
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            self.models_loaded = False
            raise
    
    def _initialize_llm(self):
        """Initialize LLM service for enhanced classification"""
        try:
            if self.llm_enabled:
                logger.info("Initializing Ollama LLM service...")
                ollama_url = getattr(settings, 'OLLAMA_URL', 'https://ollama.hellodigi.id')
                self.llm_service = get_ollama_service(base_url=ollama_url)
                
                # Check LLM health
                health = self.llm_service.health_check()
                if health['status'] == 'healthy':
                    logger.info(f"LLM service initialized successfully - Model: {health['model']}")
                else:
                    logger.warning(f"LLM service unhealthy: {health.get('error', 'Unknown error')}")
                    self.llm_enabled = False
            else:
                logger.info("LLM enhancement disabled")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            self.llm_enabled = False
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of all models"""
        health_data = {
            "models_loaded": self.models_loaded,
            "device": self.device,
            "models": {
                "classification": hasattr(self, 'classifier'),
                "detection": hasattr(self, 'detector'),
                "segmentation": hasattr(self, 'segmentator'),
                "face_detection": hasattr(self, 'face_detector')
            },
            "face_filter": {
                "enabled": self.face_filter_enabled,
                "iou_threshold": self.face_filter_iou_threshold,
                "description": "Prevents human faces from being detected as fish"
            },
            "llm_enhancement": {
                "enabled": self.llm_enabled,
                "service_available": self.llm_service is not None
            },
            "processing_stats": dict(self.processing_stats)
        }
        
        # Add LLM health check if enabled
        if self.llm_enabled and self.llm_service:
            try:
                llm_health = self.llm_service.health_check()
                health_data["llm_enhancement"]["health"] = llm_health
            except Exception as e:
                health_data["llm_enhancement"]["health"] = {"status": "error", "error": str(e)}
        
        return health_data
    
    def configure_face_filter(self, enabled: bool = True, iou_threshold: float = 0.3):
        """
        Configure face filtering settings
        
        Args:
            enabled: Whether to enable face filtering
            iou_threshold: IoU threshold for face-fish overlap detection
        """
        self.face_filter_enabled = enabled
        self.face_filter_iou_threshold = max(0.0, min(1.0, iou_threshold))  # Clamp between 0 and 1
        
        logger.info(f"Face filter configured - Enabled: {enabled}, IoU threshold: {self.face_filter_iou_threshold}")
    
    def get_face_filter_config(self) -> Dict[str, Any]:
        """Get current face filter configuration"""
        return {
            "enabled": self.face_filter_enabled,
            "iou_threshold": self.face_filter_iou_threshold
        }
    
    def configure_llm(self, enabled: bool = True):
        """
        Configure LLM enhancement settings
        
        Args:
            enabled: Whether to enable LLM verification
        """
        self.llm_enabled = enabled
        logger.info(f"LLM enhancement configured - Enabled: {enabled}")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get current LLM configuration"""
        config = {
            "enabled": self.llm_enabled,
            "service_available": self.llm_service is not None
        }
        
        if self.llm_service:
            health = self.llm_service.health_check()
            config["health"] = health
        
        return config
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            # Decode image
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Could not decode image")
            return img_bgr
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def filter_fish_by_faces(self, fish_detections: List[Dict[str, Any]], 
                           face_detections: List[Dict[str, Any]], 
                           iou_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Filter out fish detections that overlap significantly with human face detections
        
        Args:
            fish_detections: List of fish detection results
            face_detections: List of face detection results
            iou_threshold: IoU threshold above which fish detection is considered as face
                          If None, uses self.face_filter_iou_threshold
            
        Returns:
            Filtered list of fish detections
        """
        if not self.face_filter_enabled or not face_detections:
            return fish_detections
        
        if iou_threshold is None:
            iou_threshold = self.face_filter_iou_threshold
        
        filtered_fish = []
        filtered_count = 0
        
        for fish in fish_detections:
            fish_bbox = fish['bbox']
            is_face = False
            max_iou = 0.0
            
            # Check overlap with each detected face
            for face in face_detections:
                face_bbox = face['bbox']
                iou = self.calculate_iou(fish_bbox, face_bbox)
                max_iou = max(max_iou, iou)
                
                if iou > iou_threshold:
                    is_face = True
                    filtered_count += 1
                    logger.info(f"Filtered fish detection (IoU: {iou:.3f}) - likely human face at bbox {fish_bbox}")
                    break
            
            if not is_face:
                filtered_fish.append(fish)
            else:
                logger.debug(f"Fish detection at {fish_bbox} filtered out (max IoU: {max_iou:.3f})")
        
        if filtered_count > 0:
            logger.info(f"Face filter: Removed {filtered_count} fish detections that overlapped with human faces")
        
        return filtered_fish
    
    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect fish faces in the image"""
        try:
            start_time = time.time()
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            face_boxes = self.face_detector.predict(image_rgb)[0]
            
            faces = []
            for box in face_boxes:
                faces.append({
                    "bbox": [box.x1, box.y1, box.x2, box.y2],
                    "confidence": float(box.score),
                    "area": float((box.x2 - box.x1) * (box.y2 - box.y1))
                })
            
            processing_time = time.time() - start_time
            self.processing_stats['face_detection'].append(processing_time)
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def detect_fish(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect fish in the image"""
        try:
            start_time = time.time()
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            boxes = self.detector.predict(image_rgb)[0]
            
            detections = []
            for box in boxes:
                detections.append({
                    "bbox": [box.x1, box.y1, box.x2, box.y2],
                    "confidence": float(box.score),
                    "area": float((box.x2 - box.x1) * (box.y2 - box.y1)),
                    "box_object": box  # Keep reference for further processing
                })
            
            processing_time = time.time() - start_time
            self.processing_stats['fish_detection'].append(processing_time)
            
            return detections
            
        except Exception as e:
            logger.error(f"Fish detection failed: {str(e)}")
            return []
    
    def segment_fish(self, cropped_fish_bgr: np.ndarray, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Segment a single fish"""
        try:
            start_time = time.time()
            
            # Segmentation model returns a list of FishialPolygon objects
            segmented_polygons = self.segmentator.predict(cropped_fish_bgr)
            
            if segmented_polygons and len(segmented_polygons) > 0:
                # Get the first polygon (assuming one fish per crop)
                polygon = segmented_polygons[0]
                
                # Move polygon to original position
                box = detection['box_object']
                polygon.move_to(box.x1, box.y1)
                
                processing_time = time.time() - start_time
                self.processing_stats['segmentation'].append(processing_time)
                
                # Convert polygon points to serializable format
                polygon_data = [[int(x), int(y)] for x, y in polygon.points]
                
                return {
                    "polygons": [polygon],  # Keep original for potential processing
                    "polygon_data": polygon_data,   # Serializable format for visualization
                    "processing_time": processing_time
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Fish segmentation failed: {str(e)}")
            return None
    
    def classify_fish(self, cropped_fish_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Classify a single fish"""
        try:
            start_time = time.time()
            
            # Classify fish
            classification_results = self.classifier([cropped_fish_bgr])
            
            processing_time = time.time() - start_time
            self.processing_stats['classification'].append(processing_time)
            
            # Format classification results
            formatted_results = []
            for result in classification_results[0]:  # Get first (and only) result from batch
                formatted_results.append({
                    "name": result.name,
                    "species_id": result.species_id,
                    "accuracy": float(result.accuracy),
                    "distance": float(result.distance)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Fish classification failed: {str(e)}")
            return []

    def process_detection_frame(self, image_data: bytes, include_faces: bool = False,
                                include_visualization: bool = False) -> Dict[str, Any]:
        """Lightweight processing pipeline for live object detection stream"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")

        start_time = time.time()

        try:
            image_bgr = self.preprocess_image(image_data)

            faces = []
            if include_faces:
                faces = self.detect_faces(image_bgr)

            fish_candidates = self.detect_fish(image_bgr)

            if include_faces and faces and self.face_filter_enabled:
                fish_candidates = self.filter_fish_by_faces(fish_candidates, faces)

            simplified_detections = []
            for idx, detection in enumerate(fish_candidates):
                simplified_detections.append({
                    "id": idx,
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                    "area": detection["area"]
                })

            processing_time = time.time() - start_time

            result = {
                "has_fish": len(simplified_detections) > 0,
                "fish_detections": simplified_detections,
                "faces": faces,
                "image_shape": list(image_bgr.shape),
                "processing_time": processing_time
            }

            if simplified_detections:
                confidences = [det["confidence"] for det in simplified_detections]
                result["detection_summary"] = {
                    "count": len(simplified_detections),
                    "max_confidence": max(confidences),
                    "avg_confidence": float(np.mean(confidences))
                }
            else:
                result["detection_summary"] = {
                    "count": 0,
                    "max_confidence": 0.0,
                    "avg_confidence": 0.0
                }

            if include_visualization:
                try:
                    visualization_input = {
                        "fish_detections": [
                            {
                                "bbox": det["bbox"],
                                "confidence": det["confidence"],
                                "classification": [],
                                "segmentation": {
                                    "has_segmentation": False,
                                    "polygon_data": []
                                }
                            }
                            for det in simplified_detections
                        ],
                        "faces": faces
                    }
                    visualization = draw_detection_results(image_bgr, visualization_input)
                    result["visualization_image"] = image_to_base64(visualization)
                except Exception as viz_error:
                    logger.warning(f"Failed to generate detection visualization: {viz_error}")
                    result["visualization_image"] = None

            return result

        except Exception as e:
            logger.error(f"Detection frame processing failed: {str(e)}")
            raise

    def process_image(self, image_data: bytes, include_faces: bool = True, 
                     include_segmentation: bool = True) -> Dict[str, Any]:
        """
        Complete fish recognition pipeline
        
        Args:
            image_data: Raw image bytes
            include_faces: Whether to detect fish faces
            include_segmentation: Whether to perform segmentation
            
        Returns:
            Complete recognition results
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            image_bgr = self.preprocess_image(image_data)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            results = {
                "image_shape": image_bgr.shape,
                "fish_detections": [],
                "faces": [],
                "processing_time": {},
                "total_processing_time": 0
            }
            
            # Detect faces if requested
            if include_faces:
                results["faces"] = self.detect_faces(image_bgr)
                logger.info(f"Detected {len(results['faces'])} human faces")
            
            # Detect fish
            fish_detections = self.detect_fish(image_bgr)
            logger.info(f"Detected {len(fish_detections)} initial fish candidates")
            
            # Filter fish detections to remove human faces
            if include_faces and results["faces"] and self.face_filter_enabled:
                original_count = len(fish_detections)
                fish_detections = self.filter_fish_by_faces(fish_detections, results["faces"])
                filtered_count = original_count - len(fish_detections)
                if filtered_count > 0:
                    logger.info(f"Face filter: Removed {filtered_count} detections, {len(fish_detections)} fish detections remaining")
                else:
                    logger.info(f"Face filter: No detections filtered, {len(fish_detections)} fish detections remaining")
            elif not self.face_filter_enabled:
                logger.info("Face filter disabled - not filtering fish detections")
            else:
                logger.info("No faces detected - no filtering needed")
            
            # Process each detected fish
            for i, detection in enumerate(fish_detections):
                box = detection['box_object']
                cropped_fish_bgr = box.get_mask_BGR()
                cropped_fish_rgb = box.get_mask_RGB()
                
                fish_result = {
                    "id": i,
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                    "area": detection["area"],
                    "classification": [],
                    "segmentation": None
                }
                
                # Classify fish
                classification_results = self.classify_fish(cropped_fish_bgr)
                fish_result["classification"] = classification_results
                
                # LLM Enhancement - Primary identification system
                if self.llm_enabled and self.llm_service:
                    try:
                        llm_start = time.time()
                        
                        # Prepare detection info for LLM
                        detection_info = {
                            "classification": classification_results,
                            "detection_confidence": detection["confidence"],
                            "area": detection["area"]
                        }
                        
                        # Get LLM identification (primary result)
                        llm_result = self.llm_service.verify_classification(
                            cropped_fish_bgr,
                            detection_info
                        )
                        
                        if llm_result:
                            fish_result["llm_verification"] = {
                                "scientific_name": llm_result["scientific_name"],
                                "indonesian_name": llm_result["indonesian_name"],
                                "processing_time": time.time() - llm_start,
                                "raw_response": llm_result.get("llm_raw_response", "")
                            }
                            
                            # Override classification dengan LLM result sebagai output utama
                            # Tambahkan sebagai top prediction
                            fish_result["classification"].insert(0, {
                                "name": llm_result["indonesian_name"],
                                "scientific_name": llm_result["scientific_name"],
                                "accuracy": 0.95,  # High confidence for LLM result
                                "source": "llm",
                                "species_id": -1  # Special ID untuk LLM result
                            })
                            
                            logger.info(f"LLM identified fish {i}: {llm_result['indonesian_name']} ({llm_result['scientific_name']})")
                        else:
                            fish_result["llm_verification"] = None
                            logger.warning(f"LLM verification failed for fish {i}, using model prediction")
                        
                        self.processing_stats['llm_verification'].append(time.time() - llm_start)
                    except Exception as llm_error:
                        logger.error(f"LLM verification error for fish {i}: {str(llm_error)}")
                        fish_result["llm_verification"] = {"error": str(llm_error)}
                else:
                    fish_result["llm_verification"] = None
                
                # Segment fish if requested
                if include_segmentation:
                    segmentation_result = self.segment_fish(cropped_fish_bgr, detection)
                    if segmentation_result:
                        fish_result["segmentation"] = {
                            "has_segmentation": True,
                            "polygon_data": segmentation_result["polygon_data"],
                            "processing_time": segmentation_result["processing_time"]
                        }
                    else:
                        fish_result["segmentation"] = {
                            "has_segmentation": False,
                            "polygon_data": [],
                            "processing_time": 0
                        }
                
                results["fish_detections"].append(fish_result)
            
            # Calculate processing times
            total_time = time.time() - start_time
            results["total_processing_time"] = total_time
            
            # Add average processing times
            results["processing_time"] = {
                "face_detection": np.mean(self.processing_stats['face_detection'][-10:]) if self.processing_stats['face_detection'] else 0,
                "fish_detection": np.mean(self.processing_stats['fish_detection'][-10:]) if self.processing_stats['fish_detection'] else 0,
                "classification": np.mean(self.processing_stats['classification'][-10:]) if self.processing_stats['classification'] else 0,
                "segmentation": np.mean(self.processing_stats['segmentation'][-10:]) if self.processing_stats['segmentation'] else 0,
                "total": total_time
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise
    
    def process_batch(self, image_batch: List[bytes], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        results = []
        for image_data in image_batch:
            try:
                result = self.process_image(image_data, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "success": False
                })
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = {}
        for operation, times in self.processing_stats.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "mean": float(np.mean(times)),
                    "median": float(np.median(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                    "std": float(np.std(times))
                }
        return stats


# Global instance
fish_engine = None

def get_fish_engine() -> FishRecognitionEngine:
    """Get the global fish recognition engine instance"""
    global fish_engine
    if fish_engine is None:
        fish_engine = FishRecognitionEngine()
    return fish_engine
