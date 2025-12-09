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
from django.db.models import Avg

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser, FileUploadParser

from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

from .serializers import (
    ImageUploadSerializer, BatchImageUploadSerializer, RecognitionResultSerializer,
    BatchRecognitionResultSerializer, HealthCheckSerializer, PerformanceStatsSerializer,
    ModelConfigSerializer, FaceFilterConfigSerializer, LWFAdaptationSerializer
)
from .ml_models.fish_engine import get_fish_engine
from . import schema_examples
from .utils.image_utils import (
    base64_to_image, ImageQualityValidator, draw_detection_results, image_to_base64
)
from .utils.batch_utils import aggregate_species_votes
from .services.lwf_adapter import LWFEmbeddingAdapter
from .services.minio_service import get_minio_service
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
import uuid as uuid_lib


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


def _save_identification_to_db(
    image_data: bytes,
    original_image,
    fish_detection: dict,
    llm_result: dict,
    kb_context: dict = None,
    user_identifier: str = None,
    user_location: str = None
):
    """
    Save fish identification to database with MinIO storage
    
    Args:
        image_data: Image bytes (JPEG)
        original_image: Original uploaded file (optional)
        fish_detection: Detection results from AI
        llm_result: LLM verification result
        kb_context: Knowledge base context
        user_identifier: User ID or session ID
        user_location: User location
        
    Returns:
        UUID of created identification
    """
    from django.core.files.base import ContentFile
    from datetime import datetime
    from .models import FishIdentification
    
    # Generate UUID for this identification
    identification_uuid = uuid_lib.uuid4()
    
    # Prepare file paths
    date_path = datetime.now().strftime('%Y/%m/%d')
    image_filename = f"fish_images/{date_path}/{identification_uuid}.jpg"
    thumbnail_filename = f"fish_thumbnails/{date_path}/{identification_uuid}.jpg"
    
    # Get MinIO service
    minio = get_minio_service()
    
    # Upload to MinIO
    image_url = None
    thumbnail_url = None
    thumbnail_bytes = None
    
    if minio.enabled:
        # Upload original image
        image_file = BytesIO(image_data)
        image_url = minio.upload_image(image_file, image_filename)
        
        # Create and upload thumbnail
        image_file.seek(0)
        thumbnail_bytes = minio.create_thumbnail(image_file, max_size=(300, 300))
        if thumbnail_bytes:
            thumbnail_url = minio.upload_image(thumbnail_bytes, thumbnail_filename)
            logger.info(f"Uploaded thumbnail to MinIO: {thumbnail_url}")
    else:
        # Create thumbnail for local storage
        image_file = BytesIO(image_data)
        thumbnail_bytes = minio.create_thumbnail(image_file, max_size=(300, 300))
    
    # Extract scientific name - prioritize from LLM
    scientific_name = (
        llm_result.get('scientific_name') or 
        llm_result.get('name') or 
        'Unknown'
    )
    
    # Extract Indonesian name - LLM uses both 'indonesian_name' and 'name' for display
    indonesian_name = (
        llm_result.get('indonesian_name') or 
        llm_result.get('name') or 
        llm_result.get('label') or 
        llm_result.get('nama_indonesia') or
        'Unknown'
    )
    
    # Extract English name - directly from LLM result (populated from KB)
    english_name = (
        llm_result.get('english_name') or
        llm_result.get('common_name') or
        None
    )
    
    # Extract kelompok/group - directly from LLM result (populated from KB)
    kelompok = (
        llm_result.get('kelompok') or
        llm_result.get('group') or
        None
    )
    
    # Extract confidence
    confidence = (
        llm_result.get('confidence') or
        llm_result.get('accuracy') or
        fish_detection.get('confidence') or
        0.0
    )
    
    # Get KB candidates data
    kb_candidates_data = None
    if kb_context:
        if isinstance(kb_context, dict):
            similar_species = kb_context.get('similar_species', [])
            kb_candidates_data = similar_species
    
    # Log extracted data for debugging
    logger.info(f"Extracted data - Scientific: {scientific_name}, Indonesian: {indonesian_name}, "
                f"English: {english_name}, Kelompok: {kelompok}, Confidence: {confidence}")
    logger.info(f"Image URL: {image_url}, Thumbnail URL: {thumbnail_url}")
    
    # Create Django file objects only if MinIO is not enabled
    image_content_file = None
    thumbnail_content_file = None
    
    if not minio.enabled:
        # Use local filesystem storage
        image_content_file = ContentFile(image_data, name=f"{identification_uuid}.jpg")
        if thumbnail_bytes:
            thumbnail_bytes.seek(0)
            thumbnail_content_file = ContentFile(thumbnail_bytes.read(), name=f"{identification_uuid}_thumb.jpg")
    
    # Create identification record
    identification = FishIdentification.objects.create(
        id=identification_uuid,
        image=image_content_file,
        image_url=image_url,
        thumbnail=thumbnail_content_file,
        thumbnail_url=thumbnail_url,
        
        # Original AI predictions
        original_scientific_name=scientific_name,
        original_indonesian_name=indonesian_name,
        original_english_name=english_name,
        original_kelompok=kelompok,
        
        # Current (same as original initially)
        current_scientific_name=scientific_name,
        current_indonesian_name=indonesian_name,
        current_english_name=english_name,
        current_kelompok=kelompok,
        
        # Metadata
        confidence_score=min(float(confidence), 1.0),
        ai_model_version='v1.0',
        detection_box=fish_detection.get('bbox', []),
        detection_score=fish_detection.get('confidence'),
        kb_candidates=kb_candidates_data,
        
        # User info
        user_identifier=user_identifier,
        user_location=user_location,
        
        # Status
        status='pending'
    )
    
    # Update species statistics
    _update_species_statistics(identification)
    
    logger.info(f"Created fish identification: {identification_uuid}")
    return identification_uuid


logger = logging.getLogger(__name__)


@extend_schema_view(
    get=extend_schema(
        tags=['Health'],
        summary='Health check endpoint',
        description='Check the health status of the API and all AI models. Returns model loading status, device information, and processing statistics.',
        responses={
            200: OpenApiExample(
                'Success',
                value=schema_examples.HEALTH_CHECK_RESPONSE,
                response_only=True,
            ),
            500: OpenApiExample(
                'Unhealthy',
                value=schema_examples.HEALTH_CHECK_UNHEALTHY_RESPONSE,
                response_only=True,
            ),
        }
    )
)
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


@extend_schema_view(
    get=extend_schema(
        tags=['Health'],
        summary='Get performance statistics',
        description='Retrieve detailed performance metrics for all model components including average, min, and max processing times.',
        responses={
            200: OpenApiExample(
                'Success',
                value=schema_examples.PERFORMANCE_STATS_RESPONSE,
                response_only=True,
            ),
            500: {'description': 'Internal server error'},
        }
    )
)
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


@extend_schema_view(
    post=extend_schema(
        tags=['Recognition'],
        summary='Recognize fish in single image',
        description='''
        Process a single image for fish detection, classification, and recognition.
        
        Features:
        - Detects fish in the image using YOLO model
        - Classifies detected fish species
        - Performs segmentation for accurate fish boundaries
        - Detects fish faces for quality assessment
        - Uses LLM for species verification and knowledge base enrichment
        - Saves identification to database for tracking and correction
        
        The image can be provided as either:
        - Multipart file upload (`image` field)
        - Base64 encoded string (`image_base64` field)
        
        Returns comprehensive detection results including bounding boxes, confidence scores, 
        species classification, and optional visualization.
        ''',
        request=ImageUploadSerializer,
        responses={
            200: OpenApiExample(
                'Fish detected successfully',
                value=schema_examples.RECOGNITION_SUCCESS_RESPONSE,
                response_only=True,
            ),
            '200-no-fish': OpenApiExample(
                'No fish detected',
                value=schema_examples.RECOGNITION_NO_FISH_RESPONSE,
                response_only=True,
            ),
            400: OpenApiExample(
                'Validation error',
                value=schema_examples.QUALITY_VALIDATION_ERROR_RESPONSE,
                response_only=True,
            ),
            500: OpenApiExample(
                'Server error',
                value=schema_examples.RECOGNITION_ERROR_RESPONSE,
                response_only=True,
            ),
        },
        examples=[
            OpenApiExample(
                'File upload request',
                description='Upload image as multipart form data',
                value={
                    'image': '(binary file)',
                    'include_faces': True,
                    'include_segmentation': True,
                    'include_visualization': False
                },
                request_only=True,
            ),
            OpenApiExample(
                'Base64 request',
                description='Upload image as base64 string',
                value={
                    'image_base64': '/9j/4AAQSkZJRgABAQAA...',
                    'include_faces': True,
                    'include_segmentation': True,
                    'include_visualization': True
                },
                request_only=True,
            ),
        ]
    )
)
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
            
            # Save to database if fish detected
            identification_id = None
            logger.info(f"=== CHECKING IF FISH DETECTED ===")
            logger.info(f"Has fish_detections: {bool(results.get('fish_detections'))}")
            logger.info(f"Fish count: {len(results.get('fish_detections', []))}")
            
            if results.get('fish_detections') and len(results['fish_detections']) > 0:
                logger.info(f"✅ Fish detected, proceeding to save")
                try:
                    # Get first fish detection (primary)
                    fish = results['fish_detections'][0]
                    logger.info(f"Fish object keys: {list(fish.keys())}")
                    
                    # Get classification result (this always exists if fish detected)
                    classification = None
                    llm_result = None
                    kb_context = None
                    
                    logger.info(f"Has classification: {bool(fish.get('classification'))}")
                    logger.info(f"Classification length: {len(fish.get('classification', []))}")
                    
                    if fish.get('classification') and len(fish['classification']) > 0:
                        # Get classification with highest confidence/accuracy
                        classifications = fish['classification']
                        logger.info(f"All classifications: {[(c.get('name'), c.get('accuracy', 0)) for c in classifications]}")
                        
                        # Sort by accuracy/confidence (descending) and take the best one
                        classification = max(classifications, key=lambda c: c.get('accuracy', c.get('confidence', 0)))
                        llm_result = classification  # Use classification as llm_result
                        
                        # KB context might be stored in fish metadata
                        kb_context = fish.get('kb_context')
                        
                        # Debug: Log the structure
                        logger.info(f"✅ Best classification selected!")
                        logger.info(f"Classification keys: {list(classification.keys())}")
                        logger.info(f"Classification name: {classification.get('name')}")
                        logger.info(f"Classification scientific_name: {classification.get('scientific_name')}")
                        logger.info(f"Classification accuracy: {classification.get('accuracy', classification.get('confidence', 0))}")
                        if kb_context:
                            logger.info(f"KB Context keys: {list(kb_context.keys()) if isinstance(kb_context, dict) else 'Not a dict'}")
                    else:
                        logger.warning(f"❌ No classification found in fish detection")
                    
                    # Save if we have classification (always available when fish detected)
                    if classification:
                        logger.info(f"📝 Attempting to save identification to database...")
                        identification_id = _save_identification_to_db(
                            image_data=image_data,
                            original_image=data.get('image'),
                            fish_detection=fish,
                            llm_result=llm_result,
                            kb_context=kb_context,
                            user_identifier=request.data.get('user_identifier'),
                            user_location=request.data.get('user_location')
                        )
                        
                        # Add identification_id to response
                        results['identification_id'] = str(identification_id)
                        logger.info(f"✅✅✅ Saved identification to database: {identification_id}")
                        logger.info(f"✅✅✅ Added identification_id to results")
                        
                        # Add correction URL and data
                        from django.urls import reverse
                        try:
                            # Try with namespace first
                            correction_url = reverse('recognition:identification_correct', kwargs={'id': identification_id})
                        except:
                            # Fallback to without namespace
                            correction_url = f"/api/v1/identifications/{identification_id}/correct/"
                        results['correction_url'] = correction_url
                        logger.info(f"✅ Added correction_url: {correction_url}")
                        
                        # Add correction_data with the actual displayed values from classification
                        # This ensures what user sees in UI matches what they can correct
                        correction_data = {
                            'scientific_name': classification.get('scientific_name', ''),
                            'indonesian_name': classification.get('name', classification.get('indonesian_name', '')),
                            'english_name': classification.get('english_name', ''),
                            'kelompok': classification.get('kelompok', '')
                        }
                        results['correction_data'] = correction_data
                        logger.info(f"✅ Added correction_data: {correction_data}")
                        logger.info(f"📦 Final results keys: {list(results.keys())}")
                        
                    else:
                        logger.error(f"❌ Cannot save: classification is None")
                        
                except Exception as e:
                    logger.error(f"❌❌❌ Failed to save identification to database: {e}")
                    logger.exception(e)
                    # Continue without failing the request
            else:
                logger.warning(f"❌ No fish detected or empty fish_detections")
            
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


@extend_schema_view(
    post=extend_schema(
        tags=['Recognition'],
        summary='Batch fish recognition',
        description='''
        Process multiple images (up to 10) in a single request for fish recognition.
        
        Features:
        - Processes 1-10 images in parallel
        - Aggregates species detection across all images
        - Provides voting mechanism for species consensus
        - Returns individual results for each image plus aggregate summary
        
        Useful for:
        - Processing multiple angles of the same fish
        - Analyzing fish schools or groups
        - Batch processing of collected images
        ''',
        request=BatchImageUploadSerializer,
        responses={
            200: OpenApiExample(
                'Batch processing success',
                value=schema_examples.BATCH_SUCCESS_RESPONSE,
                response_only=True,
            ),
            400: {'description': 'Invalid request - check image count (1-10) and formats'},
            500: {'description': 'Server error during batch processing'},
        }
    )
)
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


@extend_schema_view(
    get=extend_schema(
        tags=['Configuration'],
        summary='Get current model configuration',
        description='Retrieve current AI model configuration settings including thresholds and device.',
        responses={
            200: OpenApiExample(
                'Current configuration',
                value=schema_examples.MODEL_CONFIG_GET_RESPONSE,
                response_only=True,
            ),
        }
    ),
    post=extend_schema(
        tags=['Configuration'],
        summary='Update model configuration',
        description='Update AI model configuration settings. Changes apply immediately to subsequent requests.',
        request=ModelConfigSerializer,
        responses={
            200: OpenApiExample(
                'Configuration updated',
                value=schema_examples.MODEL_CONFIG_RESPONSE,
                response_only=True,
            ),
            400: {'description': 'Invalid configuration values'},
        },
        examples=[
            OpenApiExample(
                'Update thresholds',
                value=schema_examples.MODEL_CONFIG_UPDATE_REQUEST,
                request_only=True,
            ),
        ]
    )
)
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


@extend_schema_view(
    get=extend_schema(
        tags=['Configuration'],
        summary='Get face filter configuration',
        description='Retrieve current face filter settings for fish detection quality control.',
        responses={200: FaceFilterConfigSerializer},
    ),
    post=extend_schema(
        tags=['Configuration'],
        summary='Update face filter configuration',
        description='Configure face detection filtering to improve fish detection quality.',
        request=FaceFilterConfigSerializer,
        responses={
            200: OpenApiExample(
                'Configuration updated',
                value=schema_examples.FACE_FILTER_CONFIG_RESPONSE,
                response_only=True,
            ),
        },
        examples=[
            OpenApiExample(
                'Update face filter',
                value=schema_examples.FACE_FILTER_CONFIG_REQUEST,
                request_only=True,
            ),
        ]
    )
)
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


@extend_schema(
    tags=['Configuration'],
    summary='Adapt model with new species data',
    description='''
    Add new species to the classification model using Learning-Without-Forgetting (LWF) technique.
    
    This endpoint allows you to extend the model's knowledge with new fish species
    without retraining from scratch. Upload images of the new species and the model
    will adapt its embeddings while preserving existing knowledge.
    ''',
    request=LWFAdaptationSerializer,
    responses={
        200: {
            'description': 'Adaptation successful',
            'examples': {
                'application/json': {
                    'species_id': 51,
                    'species_name': 'Ikan Baru',
                    'scientific_name': 'Novus piscis',
                    'new_embeddings': 15,
                    'total_embeddings': 215,
                    'majority_ratio': 0.85,
                    'centroid_shift': 0.12,
                    'augment_used': True
                }
            }
        },
        400: {'description': 'Invalid request or no valid images'},
        500: {'description': 'Adaptation failed'},
    }
)
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


# ============================================================================
# Fish Identification Management Views
# ============================================================================

from rest_framework import generics, filters
from rest_framework.decorators import action
from django.db.models import Q
from .serializers import (
    FishIdentificationSerializer,
    FishIdentificationListSerializer,
    FishIdentificationCorrectionSerializer,
    FishIdentificationStatusSerializer,
    FishIdentificationHistorySerializer,
    FishSpeciesStatisticsSerializer
)
from .services.minio_service import get_minio_service


@extend_schema_view(
    get=extend_schema(
        tags=['Identification'],
        summary='List all fish identifications',
        description='''
        Retrieve paginated list of all fish identifications with filtering options.
        
        Supports filtering by:
        - Status (pending, verified, corrected, rejected)
        - Correction status (is_corrected)
        - Date range
        - Species name
        - User identifier
        ''',
        parameters=[
            OpenApiParameter('status', type=str, description='Filter by status'),
            OpenApiParameter('is_corrected', type=bool, description='Filter by correction status'),
            OpenApiParameter('search', type=str, description='Search in species names'),
            OpenApiParameter('page', type=int, description='Page number'),
        ],
        responses={
            200: OpenApiExample(
                'Identification list',
                value=schema_examples.IDENTIFICATION_LIST_RESPONSE,
                response_only=True,
            ),
        }
    ),
    post=extend_schema(
        tags=['Identification'],
        summary='Create fish identification',
        description='Create new fish identification record (typically used internally after recognition).',
        request=FishIdentificationSerializer,
        responses={201: FishIdentificationSerializer},
    )
)
class FishIdentificationListCreateView(generics.ListCreateAPIView):
    """
    List all fish identifications or create a new one
    
    GET: List all identifications with optional filters
    POST: Create new identification (used internally after recognition)
    """
    
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = [
        'current_indonesian_name',
        'current_scientific_name',
        'original_indonesian_name',
        'original_scientific_name',
    ]
    ordering_fields = ['created_at', 'confidence_score', 'current_indonesian_name']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        if self.request.method == 'GET':
            return FishIdentificationListSerializer
        return FishIdentificationSerializer
    
    def get_queryset(self):
        from .models import FishIdentification
        queryset = FishIdentification.objects.all()
        
        # Filter by status
        status = self.request.query_params.get('status')
        if status:
            queryset = queryset.filter(status=status)
        
        # Filter by corrected status
        is_corrected = self.request.query_params.get('is_corrected')
        if is_corrected is not None:
            queryset = queryset.filter(is_corrected=is_corrected.lower() == 'true')
        
        # Filter by date range
        date_from = self.request.query_params.get('date_from')
        if date_from:
            queryset = queryset.filter(created_at__gte=date_from)
        
        date_to = self.request.query_params.get('date_to')
        if date_to:
            queryset = queryset.filter(created_at__lte=date_to)
        
        # Filter by kelompok
        kelompok = self.request.query_params.get('kelompok')
        if kelompok:
            queryset = queryset.filter(current_kelompok=kelompok)
        
        return queryset


@extend_schema_view(
    get=extend_schema(
        tags=['Identification'],
        summary='Get identification details',
        description='Retrieve detailed information about a specific fish identification including images, AI predictions, and correction history.',
        responses={
            200: OpenApiExample(
                'Identification details',
                value=schema_examples.IDENTIFICATION_DETAIL_RESPONSE,
                response_only=True,
            ),
            404: schema_examples.NOT_FOUND_ERROR_RESPONSE,
        }
    ),
    put=extend_schema(
        tags=['Identification'],
        summary='Update identification',
        description='Update fish identification details.',
        request=FishIdentificationSerializer,
        responses={200: FishIdentificationSerializer},
    ),
    patch=extend_schema(
        tags=['Identification'],
        summary='Partially update identification',
        description='Partially update fish identification details.',
        request=FishIdentificationSerializer,
        responses={200: FishIdentificationSerializer},
    ),
    delete=extend_schema(
        tags=['Identification'],
        summary='Delete identification',
        description='Delete a fish identification record.',
        responses={204: None},
    )
)
class FishIdentificationDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Retrieve, update, or delete a fish identification
    
    GET: Get identification details
    PUT/PATCH: Update identification
    DELETE: Delete identification
    """
    
    def get_queryset(self):
        from .models import FishIdentification
        return FishIdentification.objects.all()
    serializer_class = FishIdentificationSerializer
    lookup_field = 'id'


@extend_schema(
    tags=['Correction'],
    summary='Correct fish identification',
    description='''
    Submit manual correction for AI-identified fish species.
    
    Use this endpoint when the AI misidentified a fish species. Corrections are tracked
    and used to improve the model over time. The system maintains both original AI
    predictions and corrected values for accuracy tracking.
    
    All corrections are logged in the identification history for audit purposes.
    ''',
    request=FishIdentificationCorrectionSerializer,
    responses={
        200: OpenApiExample(
            'Correction successful',
            value=schema_examples.IDENTIFICATION_CORRECTION_RESPONSE,
            response_only=True,
        ),
        400: {'description': 'Invalid correction data'},
        404: schema_examples.NOT_FOUND_ERROR_RESPONSE,
    },
    examples=[
        OpenApiExample(
            'Correct species',
            value=schema_examples.IDENTIFICATION_CORRECTION_REQUEST,
            request_only=True,
        ),
    ]
)
class FishIdentificationCorrectView(APIView):
    """
    Correct a fish identification
    
    POST: Update the fish name and mark as corrected
    """
    
    def post(self, request, id):
        """
        Correct fish identification
        
        Body:
        {
            "scientific_name": "Corrected scientific name",
            "indonesian_name": "Corrected Indonesian name",
            "english_name": "Corrected English name (optional)",
            "kelompok": "Corrected kelompok (optional)",
            "notes": "Reason for correction (optional)"
        }
        """
        from .models import FishIdentification, FishIdentificationHistory
        
        logger.info(f"=== CORRECTION REQUEST ===")
        logger.info(f"ID: {id}")
        logger.info(f"Request data: {request.data}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        try:
            identification = FishIdentification.objects.get(id=id)
            logger.info(f"✅ Found identification: {identification.id}")
        except FishIdentification.DoesNotExist:
            return Response(
                {"error": "Fish identification not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = FishIdentificationCorrectionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        # Record history before making changes
        old_values = {
            'scientific_name': identification.current_scientific_name,
            'indonesian_name': identification.current_indonesian_name,
            'english_name': identification.current_english_name,
            'kelompok': identification.current_kelompok,
        }
        
        # Correct identification
        identification.correct_identification(
            scientific_name=data['scientific_name'],
            indonesian_name=data['indonesian_name'],
            english_name=data.get('english_name'),
            kelompok=data.get('kelompok'),
            notes=data.get('notes')
        )
        
        # Record history
        for field_name, old_value in old_values.items():
            new_value = getattr(identification, f'current_{field_name}') if field_name != 'kelompok' else identification.current_kelompok
            if old_value != new_value:
                FishIdentificationHistory.objects.create(
                    identification=identification,
                    field_name=field_name,
                    old_value=str(old_value),
                    new_value=str(new_value),
                    changed_by=request.data.get('user_identifier'),
                    change_reason=data.get('notes')
                )
        
        # Update species statistics
        _update_species_statistics(identification)
        
        # Return corrected identification data
        result_data = {
            'id': str(identification.id),
            'current_scientific_name': identification.current_scientific_name,
            'current_indonesian_name': identification.current_indonesian_name,
            'current_english_name': identification.current_english_name,
            'current_kelompok': identification.current_kelompok,
            'original_scientific_name': identification.original_scientific_name,
            'original_indonesian_name': identification.original_indonesian_name,
            'is_corrected': identification.is_corrected,
            'corrected_at': identification.corrected_at,
            'correction_notes': identification.correction_notes,
            'status': identification.status,
        }
        
        return Response(result_data, status=status.HTTP_200_OK)


@extend_schema(
    tags=['Correction'],
    summary='Verify correct identification',
    description='''
    Confirm that the AI correctly identified the fish species.
    
    Use this endpoint when you've reviewed the AI's identification and confirmed
    it is accurate. This helps track the model's accuracy and builds confidence
    in its predictions.
    ''',
    request=FishIdentificationStatusSerializer,
    responses={
        200: {'description': 'Verification successful', 'type': 'object'},
        404: schema_examples.NOT_FOUND_ERROR_RESPONSE,
    }
)
class FishIdentificationVerifyView(APIView):
    """
    Verify a fish identification as correct
    
    POST: Mark identification as verified
    """
    
    def post(self, request, id):
        """
        Verify that the AI identification was correct
        """
        from .models import FishIdentification
        try:
            identification = FishIdentification.objects.get(id=id)
        except FishIdentification.DoesNotExist:
            return Response(
                {"error": "Fish identification not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        identification.verify_identification()
        
        # Update species statistics
        _update_species_statistics(identification)
        
        serializer = FishIdentificationSerializer(identification)
        return Response(serializer.data, status=status.HTTP_200_OK)


@extend_schema(
    tags=['Correction'],
    summary='Reject identification',
    description='''
    Mark an identification as rejected.
    
    Use this when:
    - The image does not contain a fish
    - Image quality is too poor for accurate identification
    - The detection was a false positive
    ''',
    request=FishIdentificationStatusSerializer,
    responses={
        200: {'description': 'Rejection successful', 'type': 'object'},
        404: schema_examples.NOT_FOUND_ERROR_RESPONSE,
    }
)
class FishIdentificationRejectView(APIView):
    """
    Reject a fish identification
    
    POST: Mark identification as rejected
    """
    
    def post(self, request, id):
        """
        Reject the identification
        
        Body:
        {
            "notes": "Reason for rejection (optional)"
        }
        """
        from .models import FishIdentification
        try:
            identification = FishIdentification.objects.get(id=id)
        except FishIdentification.DoesNotExist:
            return Response(
                {"error": "Fish identification not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        notes = request.data.get('notes')
        identification.reject_identification(notes=notes)
        
        serializer = FishIdentificationSerializer(identification)
        return Response(serializer.data, status=status.HTTP_200_OK)


@extend_schema(
    tags=['Identification'],
    summary='Get identification history',
    description='Retrieve complete change history for a fish identification including all corrections and status changes.',
    responses={200: FishIdentificationHistorySerializer(many=True)}
)
class FishIdentificationHistoryView(generics.ListAPIView):
    """
    Get history of changes for a fish identification
    
    GET: List all changes
    """
    
    serializer_class = FishIdentificationHistorySerializer
    
    def get_queryset(self):
        from .models import FishIdentificationHistory
        identification_id = self.kwargs['id']
        return FishIdentificationHistory.objects.filter(
            identification_id=identification_id
        )


@extend_schema(
    tags=['Identification'],
    summary='Get species statistics',
    description='''
    Retrieve comprehensive statistics for all identified fish species.
    
    Includes:
    - Total identification count per species
    - AI accuracy rate (correct vs corrected)
    - Average confidence scores
    - First and last seen dates
    ''',
    parameters=[
        OpenApiParameter('kelompok', type=str, description='Filter by fish group'),
        OpenApiParameter('min_identifications', type=int, description='Minimum identification count'),
        OpenApiParameter('search', type=str, description='Search in species names'),
    ],
    responses={
        200: OpenApiExample(
            'Species statistics',
            value=schema_examples.SPECIES_STATISTICS_RESPONSE,
            response_only=True,
        ),
    }
)
class FishSpeciesStatisticsView(generics.ListAPIView):
    """
    Get statistics for all fish species
    
    GET: List species statistics
    """
    serializer_class = FishSpeciesStatisticsSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['indonesian_name', 'scientific_name', 'kelompok']
    ordering_fields = ['total_identifications', 'accuracy_rate', 'average_confidence']
    ordering = ['-total_identifications']
    
    def get_queryset(self):
        from .models import FishSpeciesStatistics
        queryset = FishSpeciesStatistics.objects.all()
        
        # Filter by kelompok
        kelompok = self.request.query_params.get('kelompok')
        if kelompok:
            queryset = queryset.filter(kelompok=kelompok)
        
        # Filter by minimum identifications
        min_identifications = self.request.query_params.get('min_identifications')
        if min_identifications:
            queryset = queryset.filter(total_identifications__gte=int(min_identifications))
        
        return queryset


@extend_schema(
    tags=['Health'],
    summary='Check MinIO storage health',
    description='Verify MinIO object storage connectivity and health status.',
    responses={
        200: {'description': 'MinIO is healthy', 'type': 'object'},
        500: {'description': 'MinIO is unavailable or disabled'},
    }
)
class MinIOHealthCheckView(APIView):
    """
    Check MinIO service health
    
    GET: Get MinIO health status
    """
    
    def get(self, request):
        """Get MinIO service health"""
        minio = get_minio_service()
        health = minio.health_check()
        return Response(health, status=status.HTTP_200_OK)


@extend_schema(
    tags=['Dataset'],
    summary='Export dataset for training',
    description='''
    Export fish identification dataset in various formats for model training or analysis.
    
    Supports:
    - JSON format (default)
    - CSV format (for data analysis)
    - YOLO format (for object detection training)
    
    Filters allow exporting specific subsets (verified only, date range, specific species).
    ''',
    parameters=[
        OpenApiParameter('format', type=str, enum=['json', 'csv', 'yolo'], description='Export format', default='json'),
        OpenApiParameter('status', type=str, description='Filter by status'),
        OpenApiParameter('kelompok', type=str, description='Filter by fish group'),
        OpenApiParameter('use_corrected', type=bool, description='Use corrected names', default=True),
        OpenApiParameter('date_from', type=str, description='Start date (YYYY-MM-DD)'),
        OpenApiParameter('date_to', type=str, description='End date (YYYY-MM-DD)'),
    ],
    responses={
        200: {'description': 'Dataset exported successfully'},
        400: {'description': 'Invalid parameters'},
    }
)
class ExportDatasetView(APIView):
    """
    Export identifications as training dataset
    
    GET: Export dataset in various formats
    """
    
    def get(self, request):
        """
        Export dataset for training
        
        Query Parameters:
        - format: json/csv/yolo (default: json)
        - status: Filter by status (verified/corrected)
        - kelompok: Filter by kelompok
        - min_confidence: Minimum confidence score
        - date_from, date_to: Date range
        - use_corrected: Use corrected names instead of original (default: true)
        """
        export_format = request.query_params.get('format', 'json')
        use_corrected = request.query_params.get('use_corrected', 'true').lower() == 'true'
        from .models import FishIdentification
        
        # Build query
        queryset = FishIdentification.objects.all()
        
        # Filter by status (default: only verified and corrected)
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        else:
            # Default: verified or corrected identifications
            queryset = queryset.filter(status__in=['verified', 'corrected'])
        
        # Filter by kelompok
        kelompok = request.query_params.get('kelompok')
        if kelompok:
            queryset = queryset.filter(current_kelompok=kelompok)
        
        # Filter by confidence
        min_confidence = request.query_params.get('min_confidence')
        if min_confidence:
            queryset = queryset.filter(confidence_score__gte=float(min_confidence))
        
        # Date range
        date_from = request.query_params.get('date_from')
        if date_from:
            queryset = queryset.filter(created_at__gte=date_from)
        
        date_to = request.query_params.get('date_to')
        if date_to:
            queryset = queryset.filter(created_at__lte=date_to)
        
        # Prepare export data
        dataset = []
        for identification in queryset:
            # Choose which name to use
            if use_corrected:
                scientific = identification.current_scientific_name
                indonesian = identification.current_indonesian_name
                english = identification.current_english_name
                kelompok = identification.current_kelompok
            else:
                scientific = identification.original_scientific_name
                indonesian = identification.original_indonesian_name
                english = identification.original_english_name
                kelompok = identification.original_kelompok
            
            entry = {
                'id': str(identification.id),
                'image_url': identification.image_url or request.build_absolute_uri(identification.image.url) if identification.image else None,
                'scientific_name': scientific,
                'indonesian_name': indonesian,
                'english_name': english,
                'kelompok': kelompok,
                'confidence_score': identification.confidence_score,
                'detection_box': identification.detection_box,
                'status': identification.status,
                'is_corrected': identification.is_corrected,
                'was_ai_correct': identification.was_ai_correct,
                'created_at': identification.created_at.isoformat(),
            }
            dataset.append(entry)
        
        # Format response based on export format
        if export_format == 'csv':
            import csv
            from django.http import HttpResponse
            
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="fish_dataset.csv"'
            
            if dataset:
                writer = csv.DictWriter(response, fieldnames=dataset[0].keys())
                writer.writeheader()
                writer.writerows(dataset)
            
            return response
        
        elif export_format == 'yolo':
            # YOLO format: one txt file per image with bounding boxes
            yolo_data = {
                'images': [],
                'labels': [],
                'classes': {}
            }
            
            # Build class mapping
            unique_classes = set()
            for entry in dataset:
                unique_classes.add(entry['indonesian_name'])
            
            class_to_id = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
            yolo_data['classes'] = class_to_id
            
            # Convert to YOLO format
            for entry in dataset:
                if entry['detection_box'] and len(entry['detection_box']) == 4:
                    class_id = class_to_id[entry['indonesian_name']]
                    x, y, w, h = entry['detection_box']
                    
                    # Normalize coordinates (YOLO expects 0-1 range)
                    # Assuming detection_box is [x_min, y_min, width, height]
                    # YOLO format: class_id x_center y_center width height (all normalized)
                    
                    yolo_data['images'].append({
                        'id': entry['id'],
                        'url': entry['image_url'],
                        'label': f"{class_id} {x} {y} {w} {h}"  # Simplified, needs proper normalization
                    })
            
            return Response(yolo_data, status=status.HTTP_200_OK)
        
        else:  # json (default)
            return Response({
                'total_records': len(dataset),
                'export_format': export_format,
                'use_corrected_names': use_corrected,
                'filters': {
                    'status': status_filter or 'verified,corrected',
                    'kelompok': kelompok,
                    'min_confidence': min_confidence,
                    'date_from': date_from,
                    'date_to': date_to,
                },
                'dataset': dataset
            }, status=status.HTTP_200_OK)


@extend_schema(
    tags=['Dataset'],
    summary='Get dataset statistics',
    description='''
    Retrieve comprehensive statistics about the fish identification dataset.
    
    Includes:
    - Total identifications and distribution by status
    - Species distribution and diversity metrics
    - Temporal patterns (identifications over time)
    - Quality metrics (confidence scores, correction rates)
    - User and geographic distribution
    ''',
    responses={
        200: {
            'description': 'Dataset statistics',
            'content': {
                'application/json': {
                    'example': {
                        'total_identifications': 1500,
                        'by_status': {'verified': 1200, 'corrected': 250, 'pending': 50},
                        'unique_species': 45,
                        'avg_confidence': 0.87,
                        'accuracy_rate': 0.83,
                        'correction_rate': 0.17
                    }
                }
            }
        },
    }
)
class DatasetStatisticsView(APIView):
    """
    Get dataset statistics for training
    
    GET: Get comprehensive dataset statistics
    """
    
    def get(self, request):
        """Get dataset statistics"""
        from django.db.models import Count, Avg, Q
        from .models import FishIdentification
        
        # Total identifications
        total = FishIdentification.objects.count()
        
        # By status
        by_status = FishIdentification.objects.values('status').annotate(count=Count('id'))
        
        # By kelompok
        by_kelompok = FishIdentification.objects.values('current_kelompok').annotate(
            count=Count('id'),
            avg_confidence=Avg('confidence_score')
        ).order_by('-count')
        
        # Correction statistics
        total_corrected = FishIdentification.objects.filter(is_corrected=True).count()
        correction_rate = (total_corrected / total * 100) if total > 0 else 0
        
        # AI accuracy
        ai_correct = FishIdentification.objects.filter(
            original_scientific_name=F('current_scientific_name'),
            original_indonesian_name=F('current_indonesian_name')
        ).count()
        ai_accuracy = (ai_correct / total * 100) if total > 0 else 0
        
        # Species distribution
        species_distribution = FishIdentification.objects.values(
            'current_indonesian_name', 'current_scientific_name'
        ).annotate(count=Count('id')).order_by('-count')[:20]
        
        # Confidence distribution
        confidence_ranges = {
            'very_high (0.9-1.0)': FishIdentification.objects.filter(confidence_score__gte=0.9).count(),
            'high (0.7-0.9)': FishIdentification.objects.filter(confidence_score__gte=0.7, confidence_score__lt=0.9).count(),
            'medium (0.5-0.7)': FishIdentification.objects.filter(confidence_score__gte=0.5, confidence_score__lt=0.7).count(),
            'low (<0.5)': FishIdentification.objects.filter(confidence_score__lt=0.5).count(),
        }
        
        # Training-ready dataset count
        training_ready = FishIdentification.objects.filter(
            status__in=['verified', 'corrected'],
            confidence_score__gte=0.5
        ).count()
        
        return Response({
            'total_identifications': total,
            'training_ready': training_ready,
            'by_status': list(by_status),
            'by_kelompok': list(by_kelompok)[:10],
            'correction_statistics': {
                'total_corrected': total_corrected,
                'correction_rate': round(correction_rate, 2),
            },
            'ai_accuracy': round(ai_accuracy, 2),
            'top_20_species': list(species_distribution),
            'confidence_distribution': confidence_ranges,
        }, status=status.HTTP_200_OK)


from django.db.models import F, Q
from rest_framework.pagination import PageNumberPagination


# Helper function to update species statistics
def _update_species_statistics(identification):
    """Update statistics for the species"""
    from .models import FishIdentification, FishSpeciesStatistics
    stats, created = FishSpeciesStatistics.objects.get_or_create(
        scientific_name=identification.current_scientific_name,
        defaults={
            'indonesian_name': identification.current_indonesian_name,
            'english_name': identification.current_english_name,
            'kelompok': identification.current_kelompok,
        }
    )
    
    # Recalculate statistics
    all_identifications = FishIdentification.objects.filter(
        current_scientific_name=identification.current_scientific_name
    )
    
    stats.total_identifications = all_identifications.count()
    stats.correct_identifications = all_identifications.filter(status='verified').count()
    stats.corrected_identifications = all_identifications.filter(is_corrected=True).count()
    
    # Calculate average confidence
    avg_confidence = all_identifications.aggregate(
        avg=Avg('confidence_score')
    )['avg']
    stats.average_confidence = avg_confidence or 0.0
    
    stats.save()


class FishMasterDataPagination(PageNumberPagination):
    """Pagination for master data"""
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 200


@extend_schema_view(
    get=extend_schema(
        tags=['Dataset'],
        summary='List fish master data',
        description='''
        Retrieve fish species master data from the knowledge base.
        
        This endpoint provides the reference database of known fish species
        with their Indonesian names, scientific names, and characteristics.
        ''',
        parameters=[
            OpenApiParameter('search', type=str, description='Search in species names'),
            OpenApiParameter('kelompok', type=str, description='Filter by fish group'),
            OpenApiParameter('jenis_perairan', type=str, description='Filter by water type'),
            OpenApiParameter('page', type=int, description='Page number'),
            OpenApiParameter('page_size', type=int, description='Items per page (max 200)'),
        ],
        responses={
            200: OpenApiExample(
                'Master data list',
                value=schema_examples.MASTER_DATA_RESPONSE,
                response_only=True,
            ),
        }
    ),
    post=extend_schema(
        tags=['Dataset'],
        summary='Create master data entry',
        description='Add new fish species to the master data knowledge base.',
        request={'application/json': {'type': 'object'}},
        responses={201: {'description': 'Master data created successfully'}},
    )
)
class FishMasterDataListCreateView(APIView):
    """
    GET: List all master data with filtering and search
    POST: Create new master data entry
    """
    
    def get(self, request):
        """List master data with filters"""
        from .models import FishMasterData
        from .serializers import FishMasterDataSerializer
        
        queryset = FishMasterData.objects.all()
        
        # Search by name
        search = request.query_params.get('search', '').strip()
        if search:
            queryset = queryset.filter(
                Q(species_indonesia__icontains=search) |
                Q(species_english__icontains=search) |
                Q(nama_latin__icontains=search) |
                Q(nama_daerah__icontains=search) |
                Q(kelompok__icontains=search)
            )
        
        # Filter by kelompok
        kelompok = request.query_params.get('kelompok')
        if kelompok:
            queryset = queryset.filter(kelompok=kelompok)
        
        # Filter by jenis_perairan
        jenis_perairan = request.query_params.get('jenis_perairan')
        if jenis_perairan:
            queryset = queryset.filter(jenis_perairan__icontains=jenis_perairan)
        
        # Filter by jenis_konsumsi
        jenis_konsumsi = request.query_params.get('jenis_konsumsi')
        if jenis_konsumsi:
            queryset = queryset.filter(jenis_konsumsi=jenis_konsumsi)
        
        # Filter by jenis_hias
        jenis_hias = request.query_params.get('jenis_hias')
        if jenis_hias:
            queryset = queryset.filter(jenis_hias=jenis_hias)
        
        # Filter by jenis_dilindungi
        jenis_dilindungi = request.query_params.get('jenis_dilindungi')
        if jenis_dilindungi:
            queryset = queryset.filter(jenis_dilindungi=jenis_dilindungi)
        
        # Filter by prioritas
        prioritas = request.query_params.get('prioritas')
        if prioritas:
            queryset = queryset.filter(prioritas=prioritas)
        
        # Ordering
        ordering = request.query_params.get('ordering', 'species_indonesia')
        if ordering:
            queryset = queryset.order_by(ordering)
        
        # Pagination
        paginator = FishMasterDataPagination()
        page = paginator.paginate_queryset(queryset, request)
        
        if page is not None:
            serializer = FishMasterDataSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)
        
        serializer = FishMasterDataSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        """Create new master data entry"""
        from .serializers import FishMasterDataSerializer
        
        serializer = FishMasterDataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@extend_schema_view(
    get=extend_schema(
        tags=['Dataset'],
        summary='Get master data details',
        description='Retrieve detailed information about a specific fish species from master data.',
        responses={200: {'type': 'object'}, 404: schema_examples.NOT_FOUND_ERROR_RESPONSE},
    ),
    put=extend_schema(
        tags=['Dataset'],
        summary='Update master data',
        description='Update fish species information in master data.',
        request={'application/json': {'type': 'object'}},
        responses={200: {'description': 'Updated successfully'}},
    ),
    delete=extend_schema(
        tags=['Dataset'],
        summary='Delete master data',
        description='Remove fish species from master data.',
        responses={204: None},
    )
)
class FishMasterDataDetailView(APIView):
    """
    GET: Retrieve specific master data entry
    PUT: Update master data entry
    DELETE: Delete master data entry
    """
    
    def get_object(self, pk):
        """Get master data by ID"""
        from .models import FishMasterData
        try:
            return FishMasterData.objects.get(pk=pk)
        except FishMasterData.DoesNotExist:
            return None
    
    def get(self, request, pk):
        """Retrieve master data"""
        from .serializers import FishMasterDataSerializer
        
        obj = self.get_object(pk)
        if obj is None:
            return Response(
                {'error': 'Master data not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = FishMasterDataSerializer(obj)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def put(self, request, pk):
        """Update master data"""
        from .serializers import FishMasterDataSerializer
        
        obj = self.get_object(pk)
        if obj is None:
            return Response(
                {'error': 'Master data not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = FishMasterDataSerializer(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def patch(self, request, pk):
        """Partial update master data"""
        return self.put(request, pk)
    
    def delete(self, request, pk):
        """Delete master data"""
        obj = self.get_object(pk)
        if obj is None:
            return Response(
                {'error': 'Master data not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        obj.delete()
        return Response(
            {'message': 'Master data deleted successfully'},
            status=status.HTTP_204_NO_CONTENT
        )


@extend_schema(
    tags=['Dataset'],
    summary='Get master data statistics',
    description='''
    Retrieve statistics about the fish master data knowledge base.
    
    Includes counts by:
    - Fish groups (kelompok)
    - Water types (jenis_perairan)
    - Conservation status
    - Total species count
    ''',
    responses={
        200: {
            'description': 'Master data statistics',
            'content': {
                'application/json': {
                    'example': {
                        'total_species': 150,
                        'by_kelompok': {'Ikan Pelagis': 45, 'Ikan Demersal': 38},
                        'by_water_type': {'Laut': 90, 'Air Tawar': 45, 'Air Payau': 15},
                        'protected_species': 12,
                        'ornamental_species': 28
                    }
                }
            }
        },
    }
)
class FishMasterDataStatsView(APIView):
    """Get statistics about master data"""
    
    def get(self, request):
        """Get master data statistics"""
        from .models import FishMasterData
        from django.db.models import Count
        
        total = FishMasterData.objects.count()
        
        # By kelompok
        by_kelompok = FishMasterData.objects.values('kelompok').annotate(
            count=Count('id')
        ).order_by('-count')[:10]
        
        # By jenis_perairan
        by_perairan = FishMasterData.objects.values('jenis_perairan').annotate(
            count=Count('id')
        ).order_by('-count')
        
        # By konsumsi
        by_konsumsi = FishMasterData.objects.values('jenis_konsumsi').annotate(
            count=Count('id')
        ).order_by('-count')
        
        # By hias
        by_hias = FishMasterData.objects.values('jenis_hias').annotate(
            count=Count('id')
        ).order_by('-count')
        
        # By dilindungi
        by_dilindungi = FishMasterData.objects.values('jenis_dilindungi').annotate(
            count=Count('id')
        ).order_by('-count')
        
        # By prioritas
        by_prioritas = FishMasterData.objects.values('prioritas').annotate(
            count=Count('id')
        ).order_by('-count')
        
        return Response({
            'total': total,
            'by_kelompok': list(by_kelompok),
            'by_perairan': list(by_perairan),
            'by_konsumsi': list(by_konsumsi),
            'by_hias': list(by_hias),
            'by_dilindungi': list(by_dilindungi),
            'by_prioritas': list(by_prioritas),
        }, status=status.HTTP_200_OK)

