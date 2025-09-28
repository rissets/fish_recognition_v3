"""
ASGI config for fish_recognition_api project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fish_recognition_api.settings')

# Import WebSocket consumer
from recognition.consumers.recognition_consumer import RecognitionConsumer
from recognition.consumers.detection_consumer import DetectionConsumer

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter([
        path("ws/recognition/", RecognitionConsumer.as_asgi()),
        path("ws/recognition/detection/", DetectionConsumer.as_asgi()),
    ]),
})
