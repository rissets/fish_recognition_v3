"""
URL configuration for fish_recognition_api project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from recognition.test_views import index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('recognition.urls')),
    path('', index, name='index'),  # Serve testing app at root
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
