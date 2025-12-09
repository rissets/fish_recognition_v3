"""
Serializers for Fish Recognition API
"""

from rest_framework import serializers
from django.core.files.uploadedfile import InMemoryUploadedFile
import base64


class ImageUploadSerializer(serializers.Serializer):
    """Serializer for image upload"""
    image = serializers.ImageField(required=False)
    image_base64 = serializers.CharField(required=False, allow_blank=True)
    include_faces = serializers.BooleanField(default=True, required=False)
    include_segmentation = serializers.BooleanField(default=True, required=False)
    include_visualization = serializers.BooleanField(default=False, required=False)
    
    def validate(self, data):
        """Validate that either image file or base64 is provided"""
        image = data.get('image')
        image_base64 = data.get('image_base64')
        
        if not image and not image_base64:
            raise serializers.ValidationError("Either 'image' file or 'image_base64' must be provided")
        
        if image and image_base64:
            raise serializers.ValidationError("Provide either 'image' file or 'image_base64', not both")
        
        return data


class BatchImageUploadSerializer(serializers.Serializer):
    """Serializer for batch image upload"""
    images = serializers.ListField(
        child=serializers.ImageField(),
        min_length=1,
        max_length=10,
        required=False
    )
    images_base64 = serializers.ListField(
        child=serializers.CharField(),
        min_length=1,
        max_length=10,
        required=False
    )
    include_faces = serializers.BooleanField(default=True)
    include_segmentation = serializers.BooleanField(default=True)
    include_visualization = serializers.BooleanField(default=False)
    
    def validate(self, data):
        """Validate that either images or images_base64 is provided"""
        images = data.get('images')
        images_base64 = data.get('images_base64')
        
        if not images and not images_base64:
            raise serializers.ValidationError("Either 'images' or 'images_base64' must be provided")
        
        if images and images_base64:
            raise serializers.ValidationError("Provide either 'images' or 'images_base64', not both")
        
        return data


class FishDetectionSerializer(serializers.Serializer):
    """Serializer for fish detection results"""
    id = serializers.IntegerField()
    bbox = serializers.ListField(child=serializers.FloatField())
    confidence = serializers.FloatField()
    area = serializers.FloatField()
    classification = serializers.ListField(child=serializers.DictField())
    segmentation = serializers.DictField(allow_null=True)


class FaceDetectionSerializer(serializers.Serializer):
    """Serializer for face detection results"""
    bbox = serializers.ListField(child=serializers.FloatField())
    confidence = serializers.FloatField()
    area = serializers.FloatField()


class RecognitionResultSerializer(serializers.Serializer):
    """Serializer for complete recognition results"""
    success = serializers.BooleanField()
    image_shape = serializers.ListField(child=serializers.IntegerField())
    fish_detections = FishDetectionSerializer(many=True)
    faces = FaceDetectionSerializer(many=True)
    processing_time = serializers.DictField()
    total_processing_time = serializers.FloatField()
    visualization_image = serializers.CharField(required=False, allow_null=True)
    quality_validation = serializers.DictField(required=False)
    
    # Identification and correction fields
    identification_id = serializers.UUIDField(required=False, allow_null=True)
    correction_url = serializers.CharField(required=False, allow_null=True)
    correction_data = serializers.DictField(required=False, allow_null=True)


class BatchRecognitionResultSerializer(serializers.Serializer):
    """Serializer for batch recognition results"""
    results = RecognitionResultSerializer(many=True)
    total_images = serializers.IntegerField()
    successful_images = serializers.IntegerField()
    failed_images = serializers.IntegerField()
    total_processing_time = serializers.FloatField()
    aggregate_summary = serializers.DictField(required=False)


class HealthCheckSerializer(serializers.Serializer):
    """Serializer for health check results"""
    status = serializers.CharField()
    models_loaded = serializers.BooleanField()
    device = serializers.CharField()
    models = serializers.DictField()
    processing_stats = serializers.DictField()
    uptime = serializers.FloatField()


class PerformanceStatsSerializer(serializers.Serializer):
    """Serializer for performance statistics"""
    face_detection = serializers.DictField(required=False)
    fish_detection = serializers.DictField(required=False)
    classification = serializers.DictField(required=False)
    segmentation = serializers.DictField(required=False)


class WebSocketMessageSerializer(serializers.Serializer):
    """Serializer for WebSocket messages"""
    type = serializers.CharField()
    data = serializers.DictField()
    timestamp = serializers.DateTimeField(required=False)


class CameraFrameSerializer(serializers.Serializer):
    """Serializer for camera frame data"""
    frame_data = serializers.CharField()  # Base64 encoded image
    frame_id = serializers.IntegerField(required=False)
    timestamp = serializers.DateTimeField(required=False)
    include_faces = serializers.BooleanField(default=True)
    include_segmentation = serializers.BooleanField(default=True)
    include_visualization = serializers.BooleanField(default=False, required=False)
    quality_threshold = serializers.FloatField(default=0.3)
    manual_trigger = serializers.BooleanField(default=False, required=False)


class CameraFrameBatchSerializer(serializers.Serializer):
    """Serializer for batch camera frame payload"""
    frames = serializers.ListField(
        child=serializers.CharField(),
        min_length=1,
        max_length=60
    )
    include_faces = serializers.BooleanField(default=True, required=False)
    include_segmentation = serializers.BooleanField(default=True, required=False)
    include_visualization = serializers.BooleanField(default=False, required=False)
    quality_threshold = serializers.FloatField(default=0.3, required=False)


class ModelConfigSerializer(serializers.Serializer):
    """Serializer for model configuration"""
    confidence_threshold = serializers.FloatField(min_value=0.0, max_value=1.0, required=False)
    nms_threshold = serializers.FloatField(min_value=0.0, max_value=1.0, required=False)
    segmentation_threshold = serializers.FloatField(min_value=0.0, max_value=1.0, required=False)
    device = serializers.ChoiceField(choices=['cpu', 'cuda'], required=False)
    enable_caching = serializers.BooleanField(required=False)


class FaceFilterConfigSerializer(serializers.Serializer):
    """Serializer for face filter configuration"""
    enabled = serializers.BooleanField(required=False, default=True)
    iou_threshold = serializers.FloatField(min_value=0.0, max_value=1.0, required=False, default=0.3)


class LWFAdaptationSerializer(serializers.Serializer):
    """Serializer for Learning-Without-Forgetting adaptation requests"""
    species_name = serializers.CharField()
    scientific_name = serializers.CharField(required=False, allow_blank=True)
    augment_data = serializers.BooleanField(default=True, required=False)
    images = serializers.ListField(
        child=serializers.ImageField(),
        required=False,
        allow_empty=True
    )
    images_base64 = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True
    )

    def validate(self, attrs):
        images = attrs.get('images') or []
        images_b64 = attrs.get('images_base64') or []
        if not images and not images_b64:
            raise serializers.ValidationError("Provide at least one image via 'images' or 'images_base64'")
        return attrs


# ============================================================================
# Fish Identification Serializers
# ============================================================================

class FishIdentificationSerializer(serializers.ModelSerializer):
    """Serializer for FishIdentification model"""
    
    was_ai_correct = serializers.BooleanField(read_only=True)
    image_url = serializers.URLField(read_only=True)
    
    class Meta:
        model = 'recognition.FishIdentification'
        fields = [
            'id',
            'image', 
            'image_url',
            'thumbnail',
            'original_scientific_name',
            'original_indonesian_name',
            'original_english_name',
            'original_kelompok',
            'current_scientific_name',
            'current_indonesian_name',
            'current_english_name',
            'current_kelompok',
            'confidence_score',
            'ai_model_version',
            'detection_box',
            'detection_score',
            'kb_candidates',
            'status',
            'is_corrected',
            'correction_notes',
            'user_identifier',
            'user_location',
            'created_at',
            'updated_at',
            'corrected_at',
            'was_ai_correct',
        ]
        read_only_fields = [
            'id',
            'image_url',
            'created_at',
            'updated_at',
            'was_ai_correct',
        ]


class FishIdentificationListSerializer(serializers.ModelSerializer):
    """Simplified serializer for list views"""
    
    was_ai_correct = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = 'recognition.FishIdentification'
        fields = [
            'id',
            'image_url',
            'current_indonesian_name',
            'current_scientific_name',
            'original_indonesian_name',
            'confidence_score',
            'status',
            'is_corrected',
            'created_at',
            'was_ai_correct',
        ]


class FishIdentificationCorrectionSerializer(serializers.Serializer):
    """Serializer for correcting fish identification"""
    
    scientific_name = serializers.CharField(max_length=255)
    indonesian_name = serializers.CharField(max_length=255)
    english_name = serializers.CharField(max_length=255, required=False, allow_blank=True, allow_null=True)
    kelompok = serializers.CharField(max_length=100, required=False, allow_blank=True, allow_null=True)
    notes = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    
    def validate_scientific_name(self, value):
        """Validate scientific name is not empty"""
        if not value.strip():
            raise serializers.ValidationError("Scientific name cannot be empty")
        return value.strip()
    
    def validate_indonesian_name(self, value):
        """Validate Indonesian name is not empty"""
        if not value.strip():
            raise serializers.ValidationError("Indonesian name cannot be empty")
        return value.strip()


class FishIdentificationStatusSerializer(serializers.Serializer):
    """Serializer for updating status"""
    
    status = serializers.ChoiceField(choices=['pending', 'verified', 'corrected', 'rejected'])
    notes = serializers.CharField(required=False, allow_blank=True)


class FishIdentificationHistorySerializer(serializers.ModelSerializer):
    """Serializer for identification history"""
    
    class Meta:
        model = 'recognition.FishIdentificationHistory'
        fields = [
            'id',
            'identification',
            'field_name',
            'old_value',
            'new_value',
            'changed_by',
            'changed_at',
            'change_reason',
        ]
        read_only_fields = ['id', 'changed_at']


class FishSpeciesStatisticsSerializer(serializers.ModelSerializer):
    """Serializer for species statistics"""
    
    accuracy_rate = serializers.FloatField(read_only=True)
    
    class Meta:
        model = 'recognition.FishSpeciesStatistics'
        fields = [
            'id',
            'scientific_name',
            'indonesian_name',
            'english_name',
            'kelompok',
            'total_identifications',
            'correct_identifications',
            'corrected_identifications',
            'average_confidence',
            'accuracy_rate',
            'first_seen',
            'last_seen',
        ]
        read_only_fields = ['id', 'first_seen', 'last_seen', 'accuracy_rate']
        read_only_fields = ['id', 'first_seen', 'last_seen', 'accuracy_rate']
