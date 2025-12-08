#!/bin/bash

# Fish Recognition API - Docker Setup Script

set -e

echo "🐟 Fish Recognition API - Docker Setup"
echo "======================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it with your configurations."
    echo ""
else
    echo "✅ .env file already exists."
    echo ""
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check services status
echo ""
echo "📊 Services status:"
docker-compose ps

echo ""
echo "🗄️  Setting up MinIO bucket..."
echo "Please do one of the following:"
echo ""
echo "Option 1 - Via MinIO Console (Recommended):"
echo "  1. Open http://localhost:9001"
echo "  2. Login with credentials from .env (default: minioadmin/minioadmin123)"
echo "  3. Create bucket named 'fish-media' (or as set in MINIO_BUCKET_NAME)"
echo "  4. Set bucket access policy as needed"
echo ""
echo "Option 2 - Via MinIO Client (mc):"
echo "  Run these commands:"
echo "    mc alias set local http://localhost:9000 minioadmin minioadmin123"
echo "    mc mb local/fish-media"
echo "    mc anonymous set download local/fish-media"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd fish_api
pip install -r requirements.txt

echo ""
echo "🔄 Running database migrations..."
python manage.py migrate

echo ""
echo "✅ Setup completed!"
echo ""
echo "🚀 To start the Django application, run:"
echo "   cd fish_api"
echo "   daphne -b 0.0.0.0 -p 8000 fish_recognition_api.asgi:application"
echo ""
echo "📍 Service URLs:"
echo "   Django API: http://localhost:8000"
echo "   MinIO Console: http://localhost:9001"
echo "   MinIO API: http://localhost:9000"
echo "   PostgreSQL: localhost:5432"
echo "   Redis: localhost:36379"
echo ""
