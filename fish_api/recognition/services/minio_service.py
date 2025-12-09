"""
MinIO Storage Service for Fish Images
"""

import os
import logging
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile

logger = logging.getLogger(__name__)


class MinIOService:
    """
    Service for managing image uploads to MinIO/S3
    """
    
    def __init__(self):
        """Initialize MinIO client"""
        self.endpoint = os.getenv('MINIO_ENDPOINT')
        self.access_key = os.getenv('MINIO_ACCESS_KEY', os.getenv('MINIO_ROOT_USER', 'minioadmin'))
        self.secret_key = os.getenv('MINIO_SECRET_KEY', os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin123'))
        self.bucket_name = os.getenv('MINIO_BUCKET_NAME', 'fish-media')
        self.use_ssl = os.getenv('MINIO_USE_SSL', 'False').lower() == 'true'
        
        self.enabled = bool(self.endpoint)
        
        if self.enabled:
            self.client = boto3.client(
                's3',
                endpoint_url=f"http{'s' if self.use_ssl else ''}://{self.endpoint}",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=boto3.session.Config(signature_version='s3v4'),
                region_name='us-east-1'
            )
            self._ensure_bucket_exists()
        else:
            self.client = None
            logger.warning("MinIO is not configured. Image storage will use local filesystem.")
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"MinIO bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    self.client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created MinIO bucket: {self.bucket_name}")
                    
                    # Set bucket policy to allow public read (optional)
                    # Uncomment if you want public access
                    # self._set_public_policy()
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
            else:
                logger.error(f"Error checking bucket: {e}")
    
    def _set_public_policy(self):
        """Set bucket policy to allow public read access (optional)"""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"]
                }
            ]
        }
        
        try:
            import json
            self.client.put_bucket_policy(
                Bucket=self.bucket_name,
                Policy=json.dumps(policy)
            )
            logger.info(f"Set public read policy for bucket: {self.bucket_name}")
        except ClientError as e:
            logger.error(f"Failed to set bucket policy: {e}")
    
    def upload_image(self, image_file, object_name: str, content_type: str = 'image/jpeg') -> Optional[str]:
        """
        Upload image to MinIO
        
        Args:
            image_file: File-like object or InMemoryUploadedFile
            object_name: Object key/path in bucket
            content_type: MIME type
            
        Returns:
            URL to uploaded image or None if failed
        """
        if not self.enabled:
            logger.warning("MinIO not enabled, skipping upload")
            return None
        
        try:
            # Reset file pointer if needed
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            
            # Upload to MinIO
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=image_file,
                ContentType=content_type
            )
            
            # Generate URL
            url = f"http{'s' if self.use_ssl else ''}://{self.endpoint}/{self.bucket_name}/{object_name}"
            logger.info(f"Uploaded image to MinIO: {url}")
            
            return url
            
        except ClientError as e:
            logger.error(f"Failed to upload to MinIO: {e}")
            return None
    
    def create_thumbnail(self, image_file, max_size: Tuple[int, int] = (300, 300)) -> Optional[BytesIO]:
        """
        Create thumbnail from image
        
        Args:
            image_file: Original image file
            max_size: Maximum dimensions (width, height)
            
        Returns:
            BytesIO object containing thumbnail or None
        """
        try:
            # Reset file pointer
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            
            # Open image
            img = Image.open(image_file)
            
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Create thumbnail
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to BytesIO
            thumb_io = BytesIO()
            img.save(thumb_io, format='JPEG', quality=85)
            thumb_io.seek(0)
            
            return thumb_io
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return None
    
    def get_presigned_url(self, object_name: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for temporary access
        
        Args:
            object_name: Object key in bucket
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL or None
        """
        if not self.enabled:
            return None
        
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def delete_image(self, object_name: str) -> bool:
        """
        Delete image from MinIO
        
        Args:
            object_name: Object key in bucket
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_name)
            logger.info(f"Deleted image from MinIO: {object_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete from MinIO: {e}")
            return False
    
    def health_check(self) -> dict:
        """
        Check MinIO service health
        
        Returns:
            Health status dictionary
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "MinIO is not configured"
            }
        
        try:
            # Try to list buckets
            response = self.client.list_buckets()
            return {
                "status": "healthy",
                "endpoint": self.endpoint,
                "bucket": self.bucket_name,
                "total_buckets": len(response.get('Buckets', []))
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_minio_service = None


def get_minio_service() -> MinIOService:
    """Get or create MinIO service singleton"""
    global _minio_service
    if _minio_service is None:
        _minio_service = MinIOService()
    return _minio_service
