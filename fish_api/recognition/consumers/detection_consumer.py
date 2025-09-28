"""WebSocket consumer that powers the lightweight live detection stream."""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict

import cv2
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from ..ml_models.fish_engine import get_fish_engine
from ..serializers import CameraFrameSerializer
from ..utils.image_utils import ImageQualityValidator, base64_to_image

logger = logging.getLogger(__name__)


class DetectionConsumer(AsyncWebsocketConsumer):
    """Handle continuous camera frames for fast fish object detection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_active = False
        self.last_processing_time = 0.0
        self.frame_count = 0
        self.min_processing_interval = 0.25
        self.quality_threshold = 0.2

        self.client_settings = {
            "include_faces": False,
            "include_visualization": False,
            "auto_process": True
        }

        self.session_stats: Dict[str, Any] = {
            "frames_received": 0,
            "frames_processed": 0,
            "frames_skipped": 0,
            "total_processing_time": 0.0,
            "session_start": None
        }

    async def connect(self):
        try:
            await self.accept()
            self.session_stats["session_start"] = datetime.now()

            await self.send_message(
                "detection_ready",
                {
                    "message": "Connected to fish detection stream",
                    "channel": self.channel_name,
                    "settings": self.client_settings,
                    "session_id": str(int(time.time()))
                },
            )
            logger.info("Detection WebSocket connected: %s", self.channel_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to establish detection WebSocket: %s", exc)
            await self.close()

    async def disconnect(self, close_code):
        logger.info("Detection WebSocket disconnected [%s]: %s", self.channel_name, close_code)

    async def receive(self, text_data: str):
        try:
            payload = json.loads(text_data)
        except json.JSONDecodeError as exc:
            await self.send_message(
                "error",
                {
                    "message": "Invalid JSON payload",
                    "error": str(exc)
                },
            )
            return

        message_type = payload.get("type")
        data = payload.get("data", {})

        if message_type == "camera_frame":
            await self.handle_detection_frame(data)
        elif message_type == "settings_update":
            await self.handle_settings_update(data)
        elif message_type == "get_stats":
            await self.send_session_stats()
        elif message_type == "ping":
            await self.send_message("pong", {"timestamp": datetime.now().isoformat()})
        else:
            await self.send_message(
                "error",
                {
                    "message": f"Unsupported message type: {message_type}",
                    "supported_types": ["camera_frame", "settings_update", "get_stats", "ping"],
                },
            )

    async def handle_detection_frame(self, frame_data: Dict[str, Any]):
        self.session_stats["frames_received"] += 1

        serializer = CameraFrameSerializer(data=frame_data)
        if not serializer.is_valid():
            await self.send_message(
                "frame_error",
                {
                    "message": "Invalid frame data",
                    "errors": serializer.errors
                },
            )
            return

        validated = serializer.validated_data
        manual_trigger = validated.get("manual_trigger", False)

        if not await self.should_process_frame(
            enforce_interval=not manual_trigger,
            respect_auto=not manual_trigger
        ):
            self.session_stats["frames_skipped"] += 1
            await self.send_message(
                "frame_skipped",
                {
                    "reason": "Processing too frequent",
                    "last_processing_time": self.last_processing_time
                },
            )
            return

        include_faces = validated.get("include_faces", self.client_settings["include_faces"])
        include_visualization = validated.get(
            "include_visualization", self.client_settings["include_visualization"]
        )

        quality_threshold = validated.get("quality_threshold", self.quality_threshold)
        frame_id = validated.get("frame_id", self.frame_count)

        self.processing_active = True
        start_time = time.time()

        try:
            image_bgr = await database_sync_to_async(base64_to_image)(validated["frame_data"])
            _, buffer = cv2.imencode(".jpg", image_bgr)
            image_bytes = buffer.tobytes()

            validator = ImageQualityValidator(min_quality_score=quality_threshold)
            validation_result = await database_sync_to_async(validator.validate)(image_bytes)
            quality_ok = validation_result["valid"]

            if not validation_result["valid"]:
                await self.send_message(
                    "quality_warning",
                    {
                        "message": "Image quality below threshold",
                        "validation": validation_result
                    },
                )

            engine = await database_sync_to_async(get_fish_engine)()
            detection_result = await database_sync_to_async(engine.process_detection_frame)(
                image_data=image_bytes,
                include_faces=include_faces,
                include_visualization=include_visualization,
            )

            processing_time = detection_result.get("processing_time", time.time() - start_time)
            self.session_stats["frames_processed"] += 1
            self.session_stats["total_processing_time"] += processing_time

            await self.send_message(
                "detection_result",
                {
                    "frame_id": frame_id,
                    "processing_time": processing_time,
                    "has_fish": detection_result.get("has_fish", False),
                    "detections": detection_result.get("fish_detections", []),
                    "faces": detection_result.get("faces", []),
                    "detection_summary": detection_result.get("detection_summary", {}),
                    "visualization_image": detection_result.get("visualization_image"),
                    "guidance": self._guidance_from_detection(detection_result),
                    "image_shape": detection_result.get("image_shape"),
                },
            )

            summary = detection_result.get("detection_summary", {}) or {}
            logger.info(
                "Detection result - frame=%s has_fish=%s count=%s max_conf=%.3f avg_conf=%.3f processing=%.3fs quality_ok=%s",
                frame_id,
                detection_result.get("has_fish", False),
                summary.get("count", 0),
                summary.get("max_confidence", 0.0),
                summary.get("avg_confidence", 0.0),
                processing_time,
                quality_ok,
            )

            if self.session_stats["frames_processed"] % 20 == 0:
                await self.send_session_stats()

        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.error("Detection frame processing failed: %s", exc)
            await self.send_message(
                "frame_error",
                {
                    "message": "Failed to process detection frame",
                    "error": str(exc)
                },
            )
        finally:
            self.processing_active = False
            self.last_processing_time = time.time()
            self.frame_count += 1

    async def should_process_frame(self, *, enforce_interval: bool = True, respect_auto: bool = True) -> bool:
        if self.processing_active:
            return False

        current_time = time.time()
        if enforce_interval and current_time - self.last_processing_time < self.min_processing_interval:
            return False

        if respect_auto and not self.client_settings.get("auto_process", True):
            return False

        return True

    async def handle_settings_update(self, settings_data: Dict[str, Any]):
        for key, value in settings_data.items():
            if key in self.client_settings:
                self.client_settings[key] = value

        if "quality_threshold" in settings_data:
            self.quality_threshold = float(settings_data["quality_threshold"])

        if "min_processing_interval" in settings_data:
            self.min_processing_interval = float(settings_data["min_processing_interval"])

        await self.send_message(
            "settings_updated",
            {
                "message": "Detection settings updated",
                "current_settings": self.client_settings,
                "quality_threshold": self.quality_threshold,
                "min_processing_interval": self.min_processing_interval
            },
        )

    async def send_session_stats(self):
        session_start = self.session_stats.get("session_start")
        duration = 0.0
        if session_start:
            duration = (datetime.now() - session_start).total_seconds()

        processed = self.session_stats["frames_processed"] or 1
        avg_processing = self.session_stats["total_processing_time"] / processed

        await self.send_message(
            "session_stats",
            {
                "session_duration": duration,
                "frames_received": self.session_stats["frames_received"],
                "frames_processed": self.session_stats["frames_processed"],
                "frames_skipped": self.session_stats["frames_skipped"],
                "avg_processing_time": avg_processing,
            },
        )

    async def send_message(self, message_type: str, data: Dict[str, Any]):
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        await self.send(text_data=json.dumps(message))

    def _guidance_from_detection(self, detection_result: Dict[str, Any]) -> str:
        if detection_result.get("has_fish"):
            summary = detection_result.get("detection_summary", {})
            max_conf = summary.get("max_confidence", 0.0)
            if max_conf >= 0.7:
                return "Ikan terdeteksi jelas. Tahan kamera dan tekan tombol foto."
            return "Ikan terdeteksi. Stabilkan kamera untuk hasil terbaik."
        return "Arahkan kamera ke ikan hingga terdeteksi."
