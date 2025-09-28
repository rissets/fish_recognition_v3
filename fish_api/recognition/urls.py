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
    
    # Recognition endpoints
    path('recognize/', views.RecognizeImageView.as_view(), name='recognize_image'),
    path('recognize/batch/', views.RecognizeBatchView.as_view(), name='recognize_batch'),
    path('embedding/lwf/', views.LWFEmbeddingAdaptationView.as_view(), name='embedding_lwf'),
    
    # Configuration endpoints
    path('config/', views.ModelConfigView.as_view(), name='model_config'),
    path('config/face-filter/', views.FaceFilterConfigView.as_view(), name='face_filter_config'),
]
