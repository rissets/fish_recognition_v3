"""
WebSocket Consumer for Real-time Fish Recognition
Handles live camera feed processing with focus on accuracy over speed
"""

import json
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings

from ..ml_models.fish_engine import get_fish_engine
from ..utils.image_utils import base64_to_image, ImageQualityValidator, draw_detection_results, image_to_base64
from ..serializers import (
    CameraFrameSerializer,
    WebSocketMessageSerializer,
    CameraFrameBatchSerializer,
)

logger = logging.getLogger(__name__)


class RecognitionConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time fish recognition
    Optimized for accuracy with adaptive processing
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_active = False
        self.last_processing_time = 0
        self.frame_count = 0
        self.session_stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_processing_time': 0,
            'session_start': None,
            'last_recognition_time': None
        }
        
        # Adaptive processing settings
        self.min_processing_interval = 0.5  # Minimum seconds between processing
        self.quality_threshold = 0.3
        self.adaptive_quality = True
        
        # Client settings
        self.client_settings = {
            'include_faces': True,
            'include_segmentation': True,
            'include_visualization': True,  # Enable visualization by default for live stream
            'auto_process': True,
            'processing_mode': 'accuracy'  # 'accuracy' or 'speed'
        }
    
    async def connect(self):
        """Handle WebSocket connection"""
        try:
            await self.accept()
            self.session_stats['session_start'] = datetime.now()
            
            logger.info(f"WebSocket connected: {self.channel_name}")
            
            # Send welcome message
            await self.send_message('connection_established', {
                'message': 'Connected to Fish Recognition WebSocket',
                'channel': self.channel_name,
                'settings': self.client_settings,
                'session_id': str(int(time.time()))
            })
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        logger.info(f"WebSocket disconnected: {self.channel_name}, code: {close_code}")
        
        # Send final stats if possible
        try:
            await self.send_session_stats()
        except:
            pass
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type', 'unknown')
            
            logger.debug(f"Received message type: {message_type}")
            
            if message_type == 'camera_frame':
                await self.handle_camera_frame(data.get('data', {}))

            elif message_type == 'classification_frame':
                await self.handle_classification_frame(data.get('data', {}))

            elif message_type == 'classification_batch':
                await self.handle_classification_batch(data.get('data', {}))
            
            elif message_type == 'settings_update':
                await self.handle_settings_update(data.get('data', {}))
            
            elif message_type == 'get_stats':
                await self.send_session_stats()
            
            elif message_type == 'ping':
                await self.send_message('pong', {'timestamp': datetime.now().isoformat()})
            
            else:
                await self.send_message('error', {
                    'message': f'Unknown message type: {message_type}',
                    'supported_types': ['camera_frame', 'classification_frame', 'classification_batch', 'settings_update', 'get_stats', 'ping']
                })
                
        except json.JSONDecodeError as e:
            await self.send_message('error', {
                'message': 'Invalid JSON format',
                'error': str(e)
            })
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
            await self.send_message('error', {
                'message': 'Internal server error',
                'error': str(e)
            })
    
    async def handle_camera_frame(self, frame_data: Dict[str, Any]):
        """Process incoming camera frame from continuous stream"""
        await self._process_frame(frame_data, enforce_throttle=True, source='stream')

    async def handle_classification_frame(self, frame_data: Dict[str, Any]):
        """Process captured frame requested explicitly by the client"""
        await self._process_frame(frame_data, enforce_throttle=False, source='capture')

    async def handle_classification_batch(self, batch_data: Dict[str, Any]):
        """Process a batch of captured frames for aggregated prediction"""
        serializer = CameraFrameBatchSerializer(data=batch_data)
        if not serializer.is_valid():
            await self.send_message('frame_error', {
                'message': 'Invalid batch payload',
                'errors': serializer.errors,
                'source': 'batch'
            })
            return

        validated = serializer.validated_data
        frames: List[str] = validated['frames']
        include_faces = validated.get('include_faces', self.client_settings['include_faces'])
        include_segmentation = validated.get('include_segmentation', self.client_settings['include_segmentation'])
        include_visualization = validated.get('include_visualization', self.client_settings.get('include_visualization', False))
        quality_threshold = validated.get('quality_threshold', self.quality_threshold)

        logger.info("Received classification batch with %s frames (faces=%s seg=%s viz=%s)",
                    len(frames), include_faces, include_segmentation, include_visualization)

        start_time = time.time()

        try:
            import cv2  # Local import to keep optional dependency contained

            engine = await database_sync_to_async(get_fish_engine)()
            frame_results: List[Dict[str, Any]] = []
            frame_images: Dict[int, Any] = {}
            total_quality_issues: List[Dict[str, Any]] = []

            if self.adaptive_quality:
                quality_validator = ImageQualityValidator(min_quality_score=quality_threshold)
            else:
                quality_validator = None

            for idx, frame_base64 in enumerate(frames):
                try:
                    image_bgr = await database_sync_to_async(base64_to_image)(frame_base64)
                    _, buffer = cv2.imencode('.jpg', image_bgr)
                    image_bytes = buffer.tobytes()

                    if quality_validator is not None:
                        validation_result = await database_sync_to_async(quality_validator.validate)(image_bytes)
                        if not validation_result['valid']:
                            total_quality_issues.append({
                                'frame_index': idx,
                                'validation': validation_result
                            })

                    result = await database_sync_to_async(engine.process_image)(
                        image_data=image_bytes,
                        include_faces=include_faces,
                        include_segmentation=include_segmentation
                    )

                    frame_results.append({'index': idx, 'result': result})
                    if include_visualization:
                        frame_images[idx] = image_bgr

                except Exception as frame_error:  # pragma: no cover - defensive guard
                    logger.warning("Failed to process batch frame %s: %s", idx, frame_error)

            if not frame_results:
                await self.send_message('frame_error', {
                    'message': 'No frames produced valid detections',
                    'source': 'batch'
                })
                return

            aggregate = self._aggregate_batch_results(frame_results)
            best_index = aggregate.get('top_species', {}).get('best_frame_index')

            if best_index is None:
                best_index = frame_results[0]['index']

            best_result_entry = next((entry for entry in frame_results if entry['index'] == best_index), frame_results[0])
            best_result = best_result_entry['result']

            if include_visualization and best_index in frame_images:
                try:
                    visualization = await database_sync_to_async(draw_detection_results)(frame_images[best_index], best_result)
                    best_result['visualization_image'] = await database_sync_to_async(image_to_base64)(visualization)
                except Exception as viz_error:  # pragma: no cover - visualization best effort
                    logger.warning("Failed to build visualization for batch frame %s: %s", best_index, viz_error)
                    best_result['visualization_image'] = None

            best_result['aggregate_summary'] = aggregate
            if total_quality_issues:
                best_result['quality_issues'] = total_quality_issues

            processing_time = time.time() - start_time

            logger.info(
                "Batch classification completed - frames=%s species=%s votes=%s ratio=%.3f processing=%.3fs",
                aggregate.get('frames_evaluated', len(frames)),
                aggregate.get('top_species', {}).get('name'),
                aggregate.get('top_species', {}).get('count'),
                aggregate.get('majority_ratio', 0.0),
                processing_time,
            )

            await self.send_message('recognition_result', {
                'frame_id': best_index,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'results': best_result,
                'source': 'batch',
                'batch': {
                    'size': len(frames),
                    'frames_evaluated': aggregate.get('frames_evaluated'),
                    'total_fish_detections': aggregate.get('fish_detections_total'),
                    'top_species': aggregate.get('top_species'),
                    'species_votes': aggregate.get('species_votes'),
                    'per_frame_top': aggregate.get('per_frame_top'),
                    'processing_time': processing_time
                }
            })

        except Exception as batch_error:
            logger.error("Batch classification failed: %s", batch_error)
            await self.send_message('frame_error', {
                'message': 'Batch classification failed',
                'error': str(batch_error),
                'source': 'batch'
            })

    async def _process_frame(self, frame_data: Dict[str, Any], *, enforce_throttle: bool, source: str):
        try:
            self.session_stats['frames_received'] += 1

            serializer = CameraFrameSerializer(data=frame_data)
            if not serializer.is_valid():
                await self.send_message('frame_error', {
                    'message': 'Invalid frame data',
                    'errors': serializer.errors,
                    'source': source
                })
                return

            validated_data = serializer.validated_data

            if not await self.should_process_frame(
                enforce_interval=enforce_throttle,
                respect_auto=enforce_throttle
            ):
                self.session_stats['frames_skipped'] += 1
                await self.send_message('frame_skipped', {
                    'reason': 'Processing in progress or too frequent',
                    'last_processing_time': self.last_processing_time,
                    'source': source
                })
                return

            self.processing_active = True
            frame_start_time = time.time()

            try:
                result = await self.process_frame(validated_data)

                if result:
                    self.session_stats['frames_processed'] += 1
                    processing_time = time.time() - frame_start_time
                    self.session_stats['total_processing_time'] += processing_time
                    self.session_stats['last_recognition_time'] = datetime.now()

                    await self.send_message('recognition_result', {
                        'frame_id': validated_data.get('frame_id', self.frame_count),
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat(),
                        'results': result,
                        'source': source
                    })

                    if self.session_stats['frames_processed'] % 10 == 0:
                        await self.send_session_stats()

            finally:
                self.processing_active = False
                self.last_processing_time = time.time()
                self.frame_count += 1

        except Exception as e:
            logger.error(f"Error processing %s frame: %s", source, str(e))
            await self.send_message('frame_error', {
                'message': 'Failed to process frame',
                'error': str(e),
                'source': source
            })
            self.processing_active = False
    
    async def process_frame(self, frame_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single camera frame"""
        try:
            # Convert base64 to image
            image_bgr = await database_sync_to_async(base64_to_image)(frame_data['frame_data'])
            
            # Convert to bytes for engine processing
            import cv2
            _, buffer = cv2.imencode('.jpg', image_bgr)
            image_bytes = buffer.tobytes()
            
            # Validate image quality if enabled
            if self.adaptive_quality:
                validator = ImageQualityValidator(
                    min_quality_score=frame_data.get('quality_threshold', self.quality_threshold)
                )
                validation_result = await database_sync_to_async(validator.validate)(image_bytes)
                
                if not validation_result['valid']:
                    await self.send_message('quality_warning', {
                        'message': 'Poor image quality detected',
                        'validation': validation_result
                    })
                    
                    # Skip processing if quality is too low
                    if validation_result.get('quality_score', 0) < self.quality_threshold * 0.5:
                        return None
            
            # Get fish recognition engine
            engine = await database_sync_to_async(get_fish_engine)()
            
            # Process image with engine
            recognition_results = await database_sync_to_async(engine.process_image)(
                image_data=image_bytes,
                include_faces=frame_data.get('include_faces', self.client_settings['include_faces']),
                include_segmentation=frame_data.get('include_segmentation', self.client_settings['include_segmentation'])
            )
            
            # Generate visualization if requested
            include_visualization = frame_data.get('include_visualization', self.client_settings.get('include_visualization', False))
            logger.info(f"include_visualization setting: {include_visualization}")
            logger.info(f"client_settings: {self.client_settings}")
            
            if include_visualization:
                try:
                    logger.info("Generating visualization for WebSocket frame")
                    visualization = await database_sync_to_async(draw_detection_results)(image_bgr, recognition_results)
                    recognition_results['visualization_image'] = await database_sync_to_async(image_to_base64)(visualization)
                    logger.info("Visualization generated successfully")
                except Exception as e:
                    logger.warning(f"Failed to generate visualization: {str(e)}")
                    recognition_results['visualization_image'] = None
            else:
                logger.info("Visualization not requested, skipping generation")
            
            return recognition_results
            
        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            raise
    
    async def should_process_frame(self, *, enforce_interval: bool = True, respect_auto: bool = True) -> bool:
        """Determine if current frame should be processed"""
        current_time = time.time()
        
        # Don't process if already processing
        if self.processing_active:
            return False
        
        # Respect minimum processing interval
        if enforce_interval and current_time - self.last_processing_time < self.min_processing_interval:
            return False
        
        # Auto-processing check
        if respect_auto and not self.client_settings.get('auto_process', True):
            return False
        
        return True

    def _aggregate_batch_results(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate predictions across multiple frame results"""
        total_fish = 0
        species_votes: Dict[str, Dict[str, Any]] = {}
        per_frame_top: List[Dict[str, Any]] = []

        for entry in frame_results:
            frame_index = entry['index']
            result = entry['result']
            frame_top = None

            for fish in result.get('fish_detections', []):
                total_fish += 1
                classification = fish.get('classification') or []
                if not classification:
                    continue

                top_prediction = classification[0]
                species_name = top_prediction.get('name', 'Unknown')
                accuracy = float(top_prediction.get('accuracy', 0.0))
                confidence = float(fish.get('confidence', 0.0))

                species_vote = species_votes.setdefault(species_name, {
                    'name': species_name,
                    'count': 0,
                    'best_accuracy': 0.0,
                    'best_confidence': 0.0,
                    'best_frame_index': frame_index,
                    'best_detection': None,
                })

                species_vote['count'] += 1

                if (
                    accuracy > species_vote['best_accuracy']
                    or (
                        accuracy == species_vote['best_accuracy']
                        and confidence > species_vote['best_confidence']
                    )
                ):
                    species_vote['best_accuracy'] = accuracy
                    species_vote['best_confidence'] = confidence
                    species_vote['best_frame_index'] = frame_index
                    species_vote['best_detection'] = {
                        'frame_index': frame_index,
                        'accuracy': accuracy,
                        'confidence': confidence,
                        'classification': top_prediction,
                        'bbox': fish.get('bbox'),
                    }

                if frame_top is None or accuracy > frame_top['accuracy']:
                    frame_top = {
                        'frame_index': frame_index,
                        'species': species_name,
                        'accuracy': accuracy,
                        'confidence': confidence,
                    }

            if frame_top:
                per_frame_top.append(frame_top)

        vote_list = sorted(
            species_votes.values(),
            key=lambda item: (-item['count'], -item['best_accuracy'], -item['best_confidence'])
        )

        top_species = vote_list[0] if vote_list else None
        total_votes = sum(item['count'] for item in vote_list)

        return {
            'frames_evaluated': len(frame_results),
            'fish_detections_total': total_fish,
            'species_votes': vote_list,
            'per_frame_top': per_frame_top,
            'top_species': top_species,
            'majority_ratio': (top_species['count'] / total_votes) if top_species and total_votes else 0.0,
            'total_votes': total_votes,
        }
    
    async def handle_settings_update(self, settings_data: Dict[str, Any]):
        """Update client settings"""
        try:
            # Update settings
            for key, value in settings_data.items():
                if key in self.client_settings:
                    self.client_settings[key] = value
            
            # Update adaptive settings
            if 'quality_threshold' in settings_data:
                self.quality_threshold = float(settings_data['quality_threshold'])
            
            if 'min_processing_interval' in settings_data:
                self.min_processing_interval = float(settings_data['min_processing_interval'])
            
            if 'processing_mode' in settings_data:
                mode = settings_data['processing_mode']
                if mode == 'speed':
                    self.min_processing_interval = 0.1
                    self.quality_threshold = 0.2
                elif mode == 'accuracy':
                    self.min_processing_interval = 0.5
                    self.quality_threshold = 0.3
            
            await self.send_message('settings_updated', {
                'message': 'Settings updated successfully',
                'current_settings': self.client_settings,
                'adaptive_settings': {
                    'quality_threshold': self.quality_threshold,
                    'min_processing_interval': self.min_processing_interval
                }
            })
            
        except Exception as e:
            await self.send_message('settings_error', {
                'message': 'Failed to update settings',
                'error': str(e)
            })
    
    async def send_session_stats(self):
        """Send current session statistics"""
        try:
            current_time = datetime.now()
            session_duration = (current_time - self.session_stats['session_start']).total_seconds()
            
            avg_processing_time = (
                self.session_stats['total_processing_time'] / self.session_stats['frames_processed']
                if self.session_stats['frames_processed'] > 0 else 0
            )
            
            processing_rate = (
                self.session_stats['frames_processed'] / session_duration
                if session_duration > 0 else 0
            )
            
            stats = {
                'session_duration': session_duration,
                'frames_received': self.session_stats['frames_received'],
                'frames_processed': self.session_stats['frames_processed'],
                'frames_skipped': self.session_stats['frames_skipped'],
                'processing_rate': processing_rate,
                'avg_processing_time': avg_processing_time,
                'last_recognition': (
                    self.session_stats['last_recognition_time'].isoformat()
                    if self.session_stats['last_recognition_time'] else None
                )
            }
            
            await self.send_message('session_stats', stats)
            
        except Exception as e:
            logger.error(f"Failed to send session stats: {str(e)}")
    
    async def send_message(self, message_type: str, data: Dict[str, Any]):
        """Send a formatted message to the client"""
        try:
            message = {
                'type': message_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.send(text_data=json.dumps(message))
            
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
    
    async def send_heartbeat(self):
        """Send periodic heartbeat to keep connection alive"""
        while True:
            try:
                if self.channel_layer:
                    await self.send_message('heartbeat', {
                        'timestamp': datetime.now().isoformat(),
                        'processing_active': self.processing_active
                    })
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except:
                break
