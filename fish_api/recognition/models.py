"""
Models for Fish Recognition System
"""

from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid


class FishIdentification(models.Model):
    """
    Store fish identification results with images
    Users can update the fish name if the identification is incorrect
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('verified', 'Verified'),
        ('corrected', 'Corrected by User'),
        ('rejected', 'Rejected'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Image storage
    image = models.ImageField(upload_to='fish_images/%Y/%m/%d/', max_length=500, blank=True, null=True)
    image_url = models.URLField(max_length=1000, blank=True, null=True, help_text="Full URL to image in MinIO/S3")
    thumbnail = models.ImageField(upload_to='fish_thumbnails/%Y/%m/%d/', blank=True, null=True, max_length=500)
    thumbnail_url = models.URLField(max_length=1000, blank=True, null=True, help_text="Full URL to thumbnail in MinIO/S3")
    
    # Original AI identification
    original_scientific_name = models.CharField(max_length=255, db_index=True, help_text="AI-predicted scientific name")
    original_indonesian_name = models.CharField(max_length=255, db_index=True, help_text="AI-predicted Indonesian name")
    original_english_name = models.CharField(max_length=255, blank=True, null=True)
    original_kelompok = models.CharField(max_length=100, blank=True, null=True, help_text="Original group/kelompok")
    
    # Current (possibly corrected) identification
    current_scientific_name = models.CharField(max_length=255, db_index=True, help_text="Current scientific name (may be corrected)")
    current_indonesian_name = models.CharField(max_length=255, db_index=True, help_text="Current Indonesian name (may be corrected)")
    current_english_name = models.CharField(max_length=255, blank=True, null=True)
    current_kelompok = models.CharField(max_length=100, blank=True, null=True)
    
    # AI confidence and metadata
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="AI confidence score (0-1)"
    )
    ai_model_version = models.CharField(max_length=50, blank=True, null=True)
    
    # Detection metadata
    detection_box = models.JSONField(blank=True, null=True, help_text="Bounding box coordinates [x, y, width, height]")
    detection_score = models.FloatField(blank=True, null=True)
    
    # Knowledge base context
    kb_candidates = models.JSONField(blank=True, null=True, help_text="Knowledge base candidates used")
    
    # Status and tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    is_corrected = models.BooleanField(default=False, db_index=True, help_text="True if user corrected the identification")
    correction_notes = models.TextField(blank=True, null=True, help_text="User notes on correction")
    
    # User information (optional - can be extended with User model)
    user_identifier = models.CharField(max_length=255, blank=True, null=True, db_index=True, help_text="User ID or session ID")
    user_location = models.CharField(max_length=255, blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    corrected_at = models.DateTimeField(blank=True, null=True, help_text="When user made correction")
    
    class Meta:
        db_table = 'fish_identifications'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['status', '-created_at']),
            models.Index(fields=['current_indonesian_name']),
            models.Index(fields=['original_indonesian_name']),
            models.Index(fields=['is_corrected']),
        ]
        verbose_name = 'Fish Identification'
        verbose_name_plural = 'Fish Identifications'
    
    def __str__(self):
        return f"{self.current_indonesian_name} ({self.id})"
    
    def correct_identification(self, scientific_name, indonesian_name, english_name=None, kelompok=None, notes=None):
        """
        Correct the fish identification
        """
        self.current_scientific_name = scientific_name
        self.current_indonesian_name = indonesian_name
        
        if english_name:
            self.current_english_name = english_name
        if kelompok:
            self.current_kelompok = kelompok
        
        self.is_corrected = True
        self.status = 'corrected'
        self.corrected_at = timezone.now()
        
        if notes:
            self.correction_notes = notes
        
        self.save()
    
    def verify_identification(self):
        """
        Mark identification as verified (correct)
        """
        self.status = 'verified'
        self.save()
    
    def reject_identification(self, notes=None):
        """
        Reject the identification
        """
        self.status = 'rejected'
        if notes:
            self.correction_notes = notes
        self.save()
    
    @property
    def was_ai_correct(self):
        """
        Check if AI identification was correct
        """
        return (
            self.original_scientific_name == self.current_scientific_name and
            self.original_indonesian_name == self.current_indonesian_name
        )


class FishIdentificationHistory(models.Model):
    """
    Track all changes to fish identification
    """
    
    identification = models.ForeignKey(
        FishIdentification,
        on_delete=models.CASCADE,
        related_name='history'
    )
    
    # What changed
    field_name = models.CharField(max_length=100, help_text="Field that was changed")
    old_value = models.TextField(blank=True, null=True)
    new_value = models.TextField(blank=True, null=True)
    
    # Who and when
    changed_by = models.CharField(max_length=255, blank=True, null=True)
    changed_at = models.DateTimeField(auto_now_add=True)
    
    # Change reason
    change_reason = models.TextField(blank=True, null=True)
    
    class Meta:
        db_table = 'fish_identification_history'
        ordering = ['-changed_at']
        verbose_name = 'Identification History'
        verbose_name_plural = 'Identification Histories'
    
    def __str__(self):
        return f"{self.identification.id} - {self.field_name} changed at {self.changed_at}"


class FishSpeciesStatistics(models.Model):
    """
    Aggregate statistics for each fish species
    """
    
    scientific_name = models.CharField(max_length=255, unique=True, db_index=True)
    indonesian_name = models.CharField(max_length=255, db_index=True)
    english_name = models.CharField(max_length=255, blank=True, null=True)
    kelompok = models.CharField(max_length=100, blank=True, null=True)
    
    # Statistics
    total_identifications = models.IntegerField(default=0)
    correct_identifications = models.IntegerField(default=0, help_text="Verified as correct")
    corrected_identifications = models.IntegerField(default=0, help_text="User corrected")
    
    average_confidence = models.FloatField(default=0.0)
    
    # Timestamps
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'fish_species_statistics'
        ordering = ['-total_identifications']
        verbose_name = 'Species Statistics'
        verbose_name_plural = 'Species Statistics'
    
    def __str__(self):
        return f"{self.indonesian_name} ({self.total_identifications} identifications)"
    
    @property
    def accuracy_rate(self):
        """
        Calculate accuracy rate for this species
        """
        if self.total_identifications == 0:
            return 0.0
        return (self.correct_identifications / self.total_identifications) * 100
