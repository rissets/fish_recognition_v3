"""
API Views for Fish Recognition
"""

import os
import time
import logging
from typing import List, Dict, Any
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser, FileUploadParser

from .serializers import (
    ImageUploadSerializer, BatchImageUploadSerializer, RecognitionResultSerializer,
    BatchRecognitionResultSerializer, HealthCheckSerializer, PerformanceStatsSerializer,
    ModelConfigSerializer, FaceFilterConfigSerializer, LWFAdaptationSerializer
)
from .ml_models.fish_engine import get_fish_engine
from .utils.image_utils import (
    base64_to_image, ImageQualityValidator, draw_detection_results, image_to_base64
)
from .utils.batch_utils import aggregate_species_votes
from .services.lwf_adapter import LWFEmbeddingAdapter


def _normalize_image_bytes(image_bytes: bytes) -> bytes:
    """Decode arbitrary image bytes and re-encode as JPEG."""
    # Try decoding with OpenCV first for performance
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        # Fallback to PIL for broader format support
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode('.jpg', image_bgr)
    if not success:
        raise ValueError('Failed to normalize image data')

    return buffer.tobytes()

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """Health check endpoint for monitoring API status"""
    
    def get(self, request):
        """Get API health status"""
        try:
            start_time = time.time()
            engine = get_fish_engine()
            health_data = engine.health_check()
            
            response_data = {
                "status": "healthy" if health_data["models_loaded"] else "unhealthy",
                "uptime": time.time() - start_time,
                **health_data
            }
            
            serializer = HealthCheckSerializer(data=response_data)
            if serializer.is_valid():
                return Response(serializer.data, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Serialization failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return Response({
                "status": "unhealthy",
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PerformanceStatsView(APIView):
    """Performance statistics endpoint"""
    
    def get(self, request):
        """Get detailed performance statistics"""
        try:
            engine = get_fish_engine()
            stats = engine.get_performance_stats()
            
            serializer = PerformanceStatsSerializer(data=stats)
            if serializer.is_valid():
                return Response(serializer.data, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Failed to serialize stats"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Failed to get performance stats: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class RecognizeImageView(APIView):
    """Single image recognition endpoint"""
    
    parser_classes = [MultiPartParser, JSONParser, FileUploadParser]
    
    def post(self, request):
        """Process a single image for fish recognition"""
        try:
            serializer = ImageUploadSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            # Get validated data
            data = serializer.validated_data
            include_faces = data.get('include_faces', True)
            include_segmentation = data.get('include_segmentation', True)
            include_visualization = data.get('include_visualization', False)
            
            # Get image data
            image_data = None
            if 'image' in data and data['image']:
                try:
                    image_data = _normalize_image_bytes(data['image'].read())
                except Exception as exc:
                    return Response({
                        "error": f"Unsupported image format: {exc}"
                    }, status=status.HTTP_400_BAD_REQUEST)
            elif 'image_base64' in data and data['image_base64']:
                try:
                    image_bgr = base64_to_image(data['image_base64'])
                    success, buffer = cv2.imencode('.jpg', image_bgr)
                    if not success:
                        raise ValueError('Failed to encode image')
                    image_data = buffer.tobytes()
                except Exception as e:
                    return Response({
                        "error": f"Failed to decode base64 image: {str(e)}"
                    }, status=status.HTTP_400_BAD_REQUEST)
            
            if not image_data:
                return Response({
                    "error": "No valid image data provided"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate image quality
            validator = ImageQualityValidator()
            validation_result = validator.validate(image_data)
            
            if not validation_result["valid"]:
                return Response({
                    "error": "Image validation failed",
                    "validation_errors": validation_result["errors"],
                    "quality_validation": validation_result
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Process image
            engine = get_fish_engine()
            results = engine.process_image(
                image_data=image_data,
                include_faces=include_faces,
                include_segmentation=include_segmentation
            )
            
            # Add success flag
            results["success"] = True
            results["quality_validation"] = validation_result
            
            # Generate visualization if requested
            if include_visualization:
                try:
                    import cv2
                    nparr = np.frombuffer(image_data, np.uint8)
                    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # DEBUG: Log visualization generation
                    print(f"=== VISUALIZATION DEBUG ===")
                    print(f"Include visualization: {include_visualization}")
                    print(f"Image shape: {image_bgr.shape}")
                    print(f"Results fish detections: {len(results.get('fish_detections', []))}")
                    
                    # Check segmentation data in results
                    for i, fish in enumerate(results.get('fish_detections', [])):
                        seg = fish.get('segmentation', {})
                        print(f"Fish {i}: has_segmentation={seg.get('has_segmentation')}")
                        if seg.get('polygon_data'):
                            print(f"Fish {i}: polygon_data length={len(seg['polygon_data'])}")
                    
                    visualization = draw_detection_results(image_bgr, results)
                    results["visualization_image"] = image_to_base64(visualization)
                    print(f"Visualization generated successfully, length: {len(results['visualization_image'])}")
                    print(f"=== END VISUALIZATION DEBUG ===")
                except Exception as e:
                    logger.warning(f"Failed to generate visualization: {str(e)}")
                    print(f"VISUALIZATION ERROR: {str(e)}")
                    results["visualization_image"] = None
            
            # Serialize response
            response_serializer = RecognitionResultSerializer(data=results)
            if response_serializer.is_valid():
                return Response(response_serializer.data, status=status.HTTP_200_OK)
            else:
                logger.error(f"Response serialization failed: {response_serializer.errors}")
                return Response(results, status=status.HTTP_200_OK)  # Return raw data
                
        except Exception as e:
            logger.error(f"Image recognition failed: {str(e)}")
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class RecognizeBatchView(APIView):
    """Batch image recognition endpoint"""
    
    parser_classes = [MultiPartParser, JSONParser]
    
    def post(self, request):
        """Process multiple images for fish recognition"""
        try:
            serializer = BatchImageUploadSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            # Get validated data
            data = serializer.validated_data
            include_faces = data.get('include_faces', True)
            include_segmentation = data.get('include_segmentation', True)
            include_visualization = data.get('include_visualization', False)
            
            # Prepare image data list
            image_data_list = []
            
            if 'images' in data and data['images']:
                for image_file in data['images']:
                    try:
                        normalized = _normalize_image_bytes(image_file.read())
                        image_data_list.append(normalized)
                    except Exception as exc:
                        logger.warning(f"Unsupported image format in batch: {exc}")
            
            elif 'images_base64' in data and data['images_base64']:
                for base64_string in data['images_base64']:
                    try:
                        image_bgr = base64_to_image(base64_string)
                        _, buffer = cv2.imencode('.jpg', image_bgr)
                        image_data_list.append(buffer.tobytes())
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 image: {str(e)}")
                        continue
            
            if not image_data_list:
                return Response({
                    "error": "No valid images provided"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Process batch
            start_time = time.time()
            engine = get_fish_engine()
            
            batch_results = []
            successful_count = 0
            
            for i, image_data in enumerate(image_data_list):
                try:
                    # Validate image
                    validator = ImageQualityValidator()
                    validation_result = validator.validate(image_data)
                    
                    if not validation_result["valid"]:
                        batch_results.append({
                            "success": False,
                            "image_index": i,
                            "error": "Image validation failed",
                            "quality_validation": validation_result
                        })
                        continue
                    
                    # Process image
                    result = engine.process_image(
                        image_data=image_data,
                        include_faces=include_faces,
                        include_segmentation=include_segmentation
                    )
                    
                    result["success"] = True
                    result["image_index"] = i
                    result["quality_validation"] = validation_result
                    
                    # Generate visualization if requested
                    if include_visualization:
                        try:
                            import cv2
                            nparr = np.frombuffer(image_data, np.uint8)
                            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            visualization = draw_detection_results(image_bgr, result)
                            result["visualization_image"] = image_to_base64(visualization)
                        except Exception as e:
                            logger.warning(f"Failed to generate visualization for image {i}: {str(e)}")
                            result["visualization_image"] = None
                    
                    batch_results.append(result)
                    successful_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process image {i}: {str(e)}")
                    batch_results.append({
                        "success": False,
                        "image_index": i,
                        "error": str(e)
                    })
            
            total_processing_time = time.time() - start_time
            
            aggregate_summary = aggregate_species_votes(batch_results, total_images=len(image_data_list))

            response_data = {
                "results": batch_results,
                "total_images": len(image_data_list),
                "successful_images": successful_count,
                "failed_images": len(image_data_list) - successful_count,
                "total_processing_time": total_processing_time,
                "aggregate_summary": aggregate_summary
            }
            
            # Serialize response
            response_serializer = BatchRecognitionResultSerializer(data=response_data)
            if response_serializer.is_valid():
                return Response(response_serializer.data, status=status.HTTP_200_OK)
            else:
                logger.error(f"Batch response serialization failed: {response_serializer.errors}")
                return Response(response_data, status=status.HTTP_200_OK)  # Return raw data
                
        except Exception as e:
            logger.error(f"Batch recognition failed: {str(e)}")
            return Response({
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ModelConfigView(APIView):
    """Model configuration endpoint"""
    
    def get(self, request):
        """Get current model configuration"""
        try:
            config = settings.FISH_MODEL_SETTINGS
            return Response({
                "confidence_threshold": config.get('CONFIDENCE_THRESHOLD', 0.5),
                "nms_threshold": config.get('NMS_THRESHOLD', 0.3),
                "segmentation_threshold": config.get('SEGMENTATION_THRESHOLD', 0.5),
                "device": config.get('DEVICE', 'cpu'),
                "enable_caching": config.get('ENABLE_MODEL_CACHING', True),
                "batch_size": config.get('BATCH_SIZE', 1)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request):
        """Update model configuration"""
        try:
            serializer = ModelConfigSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            # Note: In production, you might want to dynamically update model settings
            # For now, return the current configuration
            return Response({
                "message": "Configuration update requested",
                "note": "Restart required for changes to take effect",
                "requested_config": serializer.validated_data
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FaceFilterConfigView(APIView):
    """Face filter configuration endpoint"""
    
    def get(self, request):
        """Get current face filter configuration"""
        try:
            engine = get_fish_engine()
            config = engine.get_face_filter_config()
            
            return Response({
                "enabled": config["enabled"],
                "iou_threshold": config["iou_threshold"],
                "description": "Face filter prevents human faces from being detected as fish"
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        """Update face filter configuration"""
        try:
            serializer = FaceFilterConfigSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            engine = get_fish_engine()
            validated_data = serializer.validated_data
            
            # Update face filter configuration
            engine.configure_face_filter(
                enabled=validated_data.get('enabled', engine.face_filter_enabled),
                iou_threshold=validated_data.get('iou_threshold', engine.face_filter_iou_threshold)
            )
            
            # Return updated configuration
            updated_config = engine.get_face_filter_config()
            
            return Response({
                "message": "Face filter configuration updated successfully",
                "config": updated_config
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LWFEmbeddingAdaptationView(APIView):
    """Learning-Without-Forgetting embedding adaptation endpoint"""

    parser_classes = [MultiPartParser, JSONParser]

    def post(self, request):
        serializer = LWFAdaptationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        payload = serializer.validated_data
        species_name = payload['species_name']
        scientific_name = payload.get('scientific_name') or None
        augment = payload.get('augment_data', True)

        image_list: List[np.ndarray] = []

        for image_file in payload.get('images', []) or []:
            try:
                normalized_bytes = _normalize_image_bytes(image_file.read())
                nparr = np.frombuffer(normalized_bytes, np.uint8)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_bgr is not None:
                    image_list.append(image_bgr)
            except Exception as exc:
                logger.warning(f"Skipping unsupported image during LWF adaptation: {exc}")

        for image_b64 in payload.get('images_base64', []) or []:
            try:
                image_bgr = base64_to_image(image_b64)
                image_list.append(image_bgr)
            except Exception as exc:
                logger.warning(f"Skipping base64 image during LWF adaptation: {exc}")

        if not image_list:
            return Response({
                "error": "No valid images provided after decoding"
            }, status=status.HTTP_400_BAD_REQUEST)

        engine = get_fish_engine()
        config = settings.FISH_MODEL_SETTINGS
        database_path = config['CLASSIFICATION_DATABASE']
        labels_path = os.path.join(os.path.dirname(database_path), 'labels.json')

        adapter = LWFEmbeddingAdapter(
            engine=engine,
            database_path=database_path,
            labels_path=labels_path,
        )

        try:
            adaptation_result = adapter.adapt(
                species_name=species_name,
                images_bgr=image_list,
                scientific_name=scientific_name,
                augment=augment,
            )
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"LWF adaptation failed: {exc}")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        response_payload = {
            "species_id": adaptation_result.species_id,
            "species_name": adaptation_result.species_name,
            "scientific_name": adaptation_result.scientific_name,
            "new_embeddings": adaptation_result.new_embeddings,
            "total_embeddings": adaptation_result.total_embeddings,
            "majority_ratio": adaptation_result.majority_ratio,
            "centroid_shift": adaptation_result.centroid_shift,
            "augment_used": augment,
        }

        return Response(response_payload, status=status.HTTP_200_OK)
