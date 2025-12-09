"""
OpenAPI Schema Examples for Fish Recognition API
Provides comprehensive examples for all request/response scenarios
"""

# ============================================================================
# Recognition Examples
# ============================================================================

FISH_DETECTION_EXAMPLE = {
    "id": 1,
    "bbox": [100.5, 150.3, 300.8, 450.2],
    "confidence": 0.95,
    "area": 60000.0,
    "classification": [
        {
            "name": "Tuna Sirip Kuning",
            "scientific_name": "Thunnus albacares",
            "english_name": "Yellowfin Tuna",
            "kelompok": "Ikan Pelagis",
            "accuracy": 0.92,
            "confidence": 0.92
        }
    ],
    "segmentation": {
        "has_segmentation": True,
        "mask_shape": [480, 640],
        "polygon_data": [[120, 150], [280, 145], [310, 420], [125, 425]]
    }
}

FACE_DETECTION_EXAMPLE = {
    "bbox": [120.0, 180.0, 180.0, 240.0],
    "confidence": 0.88,
    "area": 3600.0
}

RECOGNITION_SUCCESS_RESPONSE = {
    "success": True,
    "image_shape": [480, 640, 3],
    "fish_detections": [FISH_DETECTION_EXAMPLE],
    "faces": [FACE_DETECTION_EXAMPLE],
    "processing_time": {
        "detection": 0.15,
        "classification": 0.25,
        "segmentation": 0.10,
        "face_detection": 0.08
    },
    "total_processing_time": 0.58,
    "visualization_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "quality_validation": {
        "valid": True,
        "score": 0.92,
        "errors": [],
        "warnings": []
    },
    "identification_id": "550e8400-e29b-41d4-a716-446655440000",
    "correction_url": "/api/v1/identifications/550e8400-e29b-41d4-a716-446655440000/correct/",
    "correction_data": {
        "scientific_name": "Thunnus albacares",
        "indonesian_name": "Tuna Sirip Kuning",
        "english_name": "Yellowfin Tuna",
        "kelompok": "Ikan Pelagis"
    }
}

RECOGNITION_NO_FISH_RESPONSE = {
    "success": True,
    "image_shape": [480, 640, 3],
    "fish_detections": [],
    "faces": [],
    "processing_time": {
        "detection": 0.12
    },
    "total_processing_time": 0.12,
    "quality_validation": {
        "valid": True,
        "score": 0.85,
        "errors": [],
        "warnings": []
    }
}

RECOGNITION_ERROR_RESPONSE = {
    "success": False,
    "error": "Failed to decode base64 image: Invalid base64 string"
}

QUALITY_VALIDATION_ERROR_RESPONSE = {
    "error": "Image validation failed",
    "validation_errors": ["Image resolution too low: 320x240, minimum required: 640x480"],
    "quality_validation": {
        "valid": False,
        "score": 0.3,
        "errors": ["Image resolution too low: 320x240, minimum required: 640x480"],
        "warnings": []
    }
}

# ============================================================================
# Batch Recognition Examples
# ============================================================================

BATCH_SUCCESS_RESPONSE = {
    "results": [
        RECOGNITION_SUCCESS_RESPONSE,
        RECOGNITION_NO_FISH_RESPONSE
    ],
    "total_images": 2,
    "successful_images": 2,
    "failed_images": 0,
    "total_processing_time": 1.15,
    "aggregate_summary": {
        "Tuna Sirip Kuning": {
            "count": 1,
            "confidence": 0.92,
            "detections": [
                {
                    "image_index": 0,
                    "detection_id": 1,
                    "confidence": 0.92
                }
            ]
        }
    }
}

# ============================================================================
# Health Check Examples
# ============================================================================

HEALTH_CHECK_RESPONSE = {
    "status": "healthy",
    "models_loaded": True,
    "device": "cpu",
    "models": {
        "classification": {"loaded": True, "path": "/models/classification/model.ckpt"},
        "detection": {"loaded": True, "path": "/models/detection/model.ts"},
        "segmentation": {"loaded": True, "path": "/models/segmentation/model.ts"},
        "face_detector": {"loaded": True, "path": "/models/face_detector/model.ts"}
    },
    "processing_stats": {
        "total_requests": 150,
        "successful_requests": 148,
        "failed_requests": 2
    },
    "uptime": 0.05
}

HEALTH_CHECK_UNHEALTHY_RESPONSE = {
    "status": "unhealthy",
    "error": "Classification model failed to load"
}

# ============================================================================
# Performance Stats Examples
# ============================================================================

PERFORMANCE_STATS_RESPONSE = {
    "face_detection": {
        "average_time": 0.08,
        "min_time": 0.05,
        "max_time": 0.15,
        "total_calls": 100
    },
    "fish_detection": {
        "average_time": 0.15,
        "min_time": 0.10,
        "max_time": 0.25,
        "total_calls": 100
    },
    "classification": {
        "average_time": 0.25,
        "min_time": 0.18,
        "max_time": 0.35,
        "total_calls": 100
    },
    "segmentation": {
        "average_time": 0.10,
        "min_time": 0.08,
        "max_time": 0.18,
        "total_calls": 100
    }
}

# ============================================================================
# Model Configuration Examples
# ============================================================================

MODEL_CONFIG_UPDATE_REQUEST = {
    "confidence_threshold": 0.7,
    "nms_threshold": 0.4,
    "segmentation_threshold": 0.6,
    "device": "cpu",
    "enable_caching": True
}

MODEL_CONFIG_RESPONSE = {
    "message": "Configuration updated successfully",
    "updated_config": MODEL_CONFIG_UPDATE_REQUEST
}

MODEL_CONFIG_GET_RESPONSE = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.3,
    "segmentation_threshold": 0.5,
    "device": "cpu",
    "enable_caching": True
}

# ============================================================================
# Face Filter Configuration Examples
# ============================================================================

FACE_FILTER_CONFIG_REQUEST = {
    "enabled": True,
    "iou_threshold": 0.3
}

FACE_FILTER_CONFIG_RESPONSE = {
    "message": "Face filter configuration updated successfully",
    "config": FACE_FILTER_CONFIG_REQUEST
}

# ============================================================================
# Identification Examples
# ============================================================================

IDENTIFICATION_DETAIL_RESPONSE = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "image": "/media/fish_images/2025/01/15/550e8400-e29b-41d4-a716-446655440000.jpg",
    "image_url": "http://minio:9000/fish-media/fish_images/2025/01/15/550e8400-e29b-41d4-a716-446655440000.jpg",
    "thumbnail": "/media/fish_thumbnails/2025/01/15/550e8400-e29b-41d4-a716-446655440000.jpg",
    "thumbnail_url": "http://minio:9000/fish-media/fish_thumbnails/2025/01/15/550e8400-e29b-41d4-a716-446655440000.jpg",
    "original_scientific_name": "Thunnus albacares",
    "original_indonesian_name": "Tuna Sirip Kuning",
    "original_english_name": "Yellowfin Tuna",
    "original_kelompok": "Ikan Pelagis",
    "current_scientific_name": "Thunnus albacares",
    "current_indonesian_name": "Tuna Sirip Kuning",
    "current_english_name": "Yellowfin Tuna",
    "current_kelompok": "Ikan Pelagis",
    "confidence_score": 0.92,
    "ai_model_version": "v1.0",
    "detection_box": [100.5, 150.3, 300.8, 450.2],
    "detection_score": 0.95,
    "kb_candidates": [
        {
            "scientific_name": "Thunnus albacares",
            "indonesian_name": "Tuna Sirip Kuning",
            "similarity": 0.92
        }
    ],
    "status": "pending",
    "is_corrected": False,
    "correction_notes": None,
    "user_identifier": "user123",
    "user_location": "Jakarta",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z",
    "corrected_at": None,
    "was_ai_correct": True
}

IDENTIFICATION_LIST_RESPONSE = {
    "count": 100,
    "next": "http://localhost:8001/api/v1/identifications/?page=2",
    "previous": None,
    "results": [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "thumbnail_url": "http://minio:9000/fish-media/fish_thumbnails/2025/01/15/550e8400-e29b-41d4-a716-446655440000.jpg",
            "current_indonesian_name": "Tuna Sirip Kuning",
            "current_scientific_name": "Thunnus albacares",
            "confidence_score": 0.92,
            "status": "pending",
            "is_corrected": False,
            "created_at": "2025-01-15T10:30:00Z"
        }
    ]
}

IDENTIFICATION_CORRECTION_REQUEST = {
    "current_scientific_name": "Thunnus alalunga",
    "current_indonesian_name": "Tuna Albacore",
    "current_english_name": "Albacore Tuna",
    "current_kelompok": "Ikan Pelagis",
    "correction_notes": "Identified by expert ichthyologist",
    "status": "verified"
}

IDENTIFICATION_CORRECTION_RESPONSE = {
    "message": "Identification corrected successfully",
    "identification": IDENTIFICATION_DETAIL_RESPONSE,
    "was_ai_correct": False
}

# ============================================================================
# Statistics Examples
# ============================================================================

SPECIES_STATISTICS_RESPONSE = {
    "count": 10,
    "next": None,
    "previous": None,
    "results": [
        {
            "scientific_name": "Thunnus albacares",
            "indonesian_name": "Tuna Sirip Kuning",
            "english_name": "Yellowfin Tuna",
            "kelompok": "Ikan Pelagis",
            "identification_count": 45,
            "avg_confidence": 0.89,
            "correct_identifications": 42,
            "corrected_identifications": 3,
            "accuracy_rate": 0.93,
            "last_identified": "2025-01-15T10:30:00Z"
        }
    ]
}

# ============================================================================
# Dataset/Master Data Examples
# ============================================================================

MASTER_DATA_RESPONSE = {
    "count": 50,
    "species": [
        {
            "scientific_name": "Thunnus albacares",
            "indonesian_name": "Tuna Sirip Kuning",
            "english_name": "Yellowfin Tuna",
            "kelompok": "Ikan Pelagis",
            "description": "Tuna sirip kuning adalah ikan pelagis besar...",
            "habitat": "Laut lepas, perairan tropis dan subtropis",
            "max_size": "2.4 m",
            "max_weight": "200 kg"
        }
    ]
}

# ============================================================================
# Error Examples
# ============================================================================

VALIDATION_ERROR_RESPONSE = {
    "image": ["This field is required."]
}

BAD_REQUEST_ERROR_RESPONSE = {
    "error": "Either 'image' file or 'image_base64' must be provided"
}

SERVER_ERROR_RESPONSE = {
    "success": False,
    "error": "Internal server error occurred during image processing"
}

NOT_FOUND_ERROR_RESPONSE = {
    "detail": "Not found."
}

UNAUTHORIZED_ERROR_RESPONSE = {
    "detail": "Authentication credentials were not provided."
}
