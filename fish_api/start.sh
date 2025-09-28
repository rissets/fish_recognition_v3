#!/bin/bash

# Fish Recognition API Startup Script

echo "🐟 Starting Fish Recognition API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists, if not copy from example
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Check if Redis is running
echo "Checking Redis connection..."
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  Redis is not running. Please start Redis server:"
    echo "   brew install redis && redis-server  # macOS"
    echo "   sudo apt-get install redis-server && redis-server  # Ubuntu"
fi

# Run migrations
echo "Running database migrations..."
python manage.py migrate

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Check if model files exist
echo "Checking model files..."
MODEL_DIR="../models"
if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️  Model directory not found at $MODEL_DIR"
    echo "   Please ensure model files are available"
fi

# Start the server
echo "🚀 Starting Django server with WebSocket support..."
echo "📱 Testing app will be available at: http://localhost:8000"
echo "📡 API endpoints available at: http://localhost:8000/api/v1/"
echo "🔌 WebSocket endpoint: ws://localhost:8000/ws/recognition/"
echo ""
echo "Press Ctrl+C to stop the server"

# Start the ASGI server (supports both HTTP and WebSocket)
daphne -b 0.0.0.0 -p 8000 fish_recognition_api.asgi:application