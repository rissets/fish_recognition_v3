"""
URL configuration for recognition app
"""

from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    # Health and status endpoints
    path('health/', views.HealthCheckView.as_view(), name='health_check'),
    path('stats/', views.PerformanceStatsView.as_view(), name='performance_stats'),
    path('minio/health/', views.MinIOHealthCheckView.as_view(), name='minio_health'),
    
    # Recognition endpoints
    path('recognize/', views.RecognizeImageView.as_view(), name='recognize_image'),
    path('recognize/batch/', views.RecognizeBatchView.as_view(), name='recognize_batch'),
    path('embedding/lwf/', views.LWFEmbeddingAdaptationView.as_view(), name='embedding_lwf'),
    
    # Configuration endpoints
    path('config/', views.ModelConfigView.as_view(), name='model_config'),
    path('config/face-filter/', views.FaceFilterConfigView.as_view(), name='face_filter_config'),
    
    # Fish Identification Management
    path('identifications/', views.FishIdentificationListCreateView.as_view(), name='identification_list'),
    path('identifications/<uuid:id>/', views.FishIdentificationDetailView.as_view(), name='identification_detail'),
    path('identifications/<uuid:id>/correct/', views.FishIdentificationCorrectView.as_view(), name='identification_correct'),
    path('identifications/<uuid:id>/verify/', views.FishIdentificationVerifyView.as_view(), name='identification_verify'),
    path('identifications/<uuid:id>/reject/', views.FishIdentificationRejectView.as_view(), name='identification_reject'),
    path('identifications/<uuid:id>/history/', views.FishIdentificationHistoryView.as_view(), name='identification_history'),
    
    # Species Statistics
    path('species/statistics/', views.FishSpeciesStatisticsView.as_view(), name='species_statistics'),
    
    # Dataset Export for Training
    path('dataset/export/', views.ExportDatasetView.as_view(), name='dataset_export'),
    path('dataset/statistics/', views.DatasetStatisticsView.as_view(), name='dataset_statistics'),
]
