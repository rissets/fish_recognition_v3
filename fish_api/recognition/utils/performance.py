"""
Performance optimization utilities for Fish Recognition API
Includes caching, image preprocessing optimizations, and batch processing
"""

import hashlib
import pickle
import time
import logging
from typing import Any, Optional, Dict, List, Tuple
from functools import wraps
import numpy as np
import cv2

from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)


def generate_image_hash(image_data: bytes) -> str:
    """Generate a hash for image data for caching purposes"""
    return hashlib.sha256(image_data).hexdigest()


def cache_result(cache_key_prefix: str, timeout: int = 3600):
    """
    Decorator for caching function results
    
    Args:
        cache_key_prefix: Prefix for cache key
        timeout: Cache timeout in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{cache_key_prefix}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            if settings.FISH_MODEL_SETTINGS.get('ENABLE_MODEL_CACHING', True):
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if settings.FISH_MODEL_SETTINGS.get('ENABLE_MODEL_CACHING', True):
                cache.set(cache_key, result, timeout)
                logger.debug(f"Cached result for {cache_key}")
            
            return result
        return wrapper
    return decorator


class ImagePreprocessor:
    """Optimized image preprocessing for faster inference"""
    
    def __init__(self):
        self.target_sizes = {
            'detection': (640, 640),
            'classification': (224, 224),
            'segmentation': (416, 416),
            'face_detection': (640, 640)
        }
    
    def optimize_image_for_inference(self, image_bgr: np.ndarray, 
                                   target_type: str = 'detection') -> np.ndarray:
        """
        Optimize image for specific inference type
        
        Args:
            image_bgr: Input image in BGR format
            target_type: Type of inference ('detection', 'classification', 'segmentation', 'face_detection')
        """
        try:
            target_size = self.target_sizes.get(target_type, (640, 640))
            
            # Resize image efficiently
            if image_bgr.shape[:2] != target_size:
                image_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Ensure contiguous array for faster processing
            if not image_bgr.flags['C_CONTIGUOUS']:
                image_bgr = np.ascontiguousarray(image_bgr)
            
            return image_bgr
            
        except Exception as e:
            logger.error(f"Image optimization failed: {str(e)}")
            return image_bgr
    
    def batch_resize_images(self, images: List[np.ndarray], 
                           target_size: Tuple[int, int]) -> List[np.ndarray]:
        """Efficiently resize multiple images"""
        resized_images = []
        for image in images:
            if image.shape[:2] != target_size:
                resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
                resized_images.append(np.ascontiguousarray(resized))
            else:
                resized_images.append(image)
        return resized_images


class ModelCache:
    """Cache for model predictions and intermediate results"""
    
    def __init__(self):
        self.memory_cache = {}
        self.max_memory_items = 100
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get_cached_prediction(self, image_hash: str, model_type: str) -> Optional[Any]:
        """Get cached prediction for an image"""
        cache_key = f"prediction:{model_type}:{image_hash}"
        
        # Try Redis cache first
        if settings.FISH_MODEL_SETTINGS.get('ENABLE_MODEL_CACHING', True):
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                self.cache_stats['hits'] += 1
                return cached_result
        
        # Try memory cache
        if cache_key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[cache_key]
        
        self.cache_stats['misses'] += 1
        return None
    
    def cache_prediction(self, image_hash: str, model_type: str, 
                        prediction: Any, timeout: int = 3600):
        """Cache a model prediction"""
        cache_key = f"prediction:{model_type}:{image_hash}"
        
        # Cache in Redis
        if settings.FISH_MODEL_SETTINGS.get('ENABLE_MODEL_CACHING', True):
            cache.set(cache_key, prediction, timeout)
        
        # Cache in memory with LRU eviction
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            self.cache_stats['evictions'] += 1
        
        self.memory_cache[cache_key] = prediction
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'memory_cache_size': len(self.memory_cache),
            'max_memory_items': self.max_memory_items,
            'hit_rate': hit_rate,
            **self.cache_stats
        }


class BatchProcessor:
    """Optimized batch processing for multiple images"""
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.preprocessor = ImagePreprocessor()
    
    def create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Split items into batches"""
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i + self.batch_size])
        return batches
    
    def process_image_batch(self, image_data_list: List[bytes], 
                           processing_func, **kwargs) -> List[Any]:
        """Process a batch of images efficiently"""
        results = []
        
        # Create batches
        batches = self.create_batches(image_data_list)
        
        for batch in batches:
            batch_results = []
            
            for image_data in batch:
                try:
                    result = processing_func(image_data, **kwargs)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
                    batch_results.append({"error": str(e), "success": False})
            
            results.extend(batch_results)
        
        return results


class PerformanceMonitor:
    """Monitor and track API performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': float('inf'),
            'requests_per_minute': 0.0,
            'last_minute_requests': [],
            'model_performance': {
                'detection': {'calls': 0, 'total_time': 0.0},
                'classification': {'calls': 0, 'total_time': 0.0},
                'segmentation': {'calls': 0, 'total_time': 0.0},
                'face_detection': {'calls': 0, 'total_time': 0.0}
            }
        }
        self.start_time = time.time()
    
    def record_request(self, processing_time: float, success: bool = True):
        """Record a request and its processing time"""
        current_time = time.time()
        
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update processing time metrics
        self.metrics['total_processing_time'] += processing_time
        self.metrics['avg_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['total_requests']
        )
        
        if processing_time > self.metrics['max_processing_time']:
            self.metrics['max_processing_time'] = processing_time
        
        if processing_time < self.metrics['min_processing_time']:
            self.metrics['min_processing_time'] = processing_time
        
        # Update requests per minute
        self.metrics['last_minute_requests'].append(current_time)
        # Remove requests older than 1 minute
        self.metrics['last_minute_requests'] = [
            t for t in self.metrics['last_minute_requests'] 
            if current_time - t <= 60
        ]
        self.metrics['requests_per_minute'] = len(self.metrics['last_minute_requests'])
    
    def record_model_performance(self, model_type: str, processing_time: float):
        """Record performance for a specific model"""
        if model_type in self.metrics['model_performance']:
            self.metrics['model_performance'][model_type]['calls'] += 1
            self.metrics['model_performance'][model_type]['total_time'] += processing_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        uptime = time.time() - self.start_time
        
        # Calculate model averages
        model_averages = {}
        for model_type, stats in self.metrics['model_performance'].items():
            if stats['calls'] > 0:
                model_averages[model_type] = {
                    'avg_time': stats['total_time'] / stats['calls'],
                    'calls': stats['calls'],
                    'total_time': stats['total_time']
                }
            else:
                model_averages[model_type] = {
                    'avg_time': 0.0,
                    'calls': 0,
                    'total_time': 0.0
                }
        
        return {
            'uptime': uptime,
            'requests': {
                'total': self.metrics['total_requests'],
                'successful': self.metrics['successful_requests'],
                'failed': self.metrics['failed_requests'],
                'success_rate': (
                    self.metrics['successful_requests'] / self.metrics['total_requests']
                    if self.metrics['total_requests'] > 0 else 0
                ),
                'requests_per_minute': self.metrics['requests_per_minute']
            },
            'processing_time': {
                'average': self.metrics['avg_processing_time'],
                'maximum': self.metrics['max_processing_time'],
                'minimum': (
                    self.metrics['min_processing_time'] 
                    if self.metrics['min_processing_time'] != float('inf') else 0
                ),
                'total': self.metrics['total_processing_time']
            },
            'model_performance': model_averages,
            'throughput': (
                self.metrics['total_requests'] / uptime
                if uptime > 0 else 0
            )
        }


# Global instances
model_cache = ModelCache()
performance_monitor = PerformanceMonitor()
batch_processor = BatchProcessor()
image_preprocessor = ImagePreprocessor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return performance_monitor


def get_model_cache() -> ModelCache:
    """Get the global model cache instance"""
    return model_cache


def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance"""
    return batch_processor


def get_image_preprocessor() -> ImagePreprocessor:
    """Get the global image preprocessor instance"""
    return image_preprocessor