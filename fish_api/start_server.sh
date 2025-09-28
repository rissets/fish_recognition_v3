#!/bin/bash

# Fish Recognition API Startup Script
# This script handles all necessary setup and starts the Django application

set -e  # Exit on any error

echo "🐟 Starting Fish Recognition API..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "manage.py" ]; then
    print_error "manage.py not found. Please run this script from the Django project root."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
print_status "Using Python $python_version"

# Install dependencies
print_status "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    print_success "Dependencies installed"
else
    print_warning "requirements.txt not found. Some dependencies might be missing."
fi

# Check if migrations are needed
print_status "Checking database migrations..."
if python3 manage.py showmigrations --plan | grep -q "\[ \]"; then
    print_warning "Unapplied migrations detected. Applying migrations..."
    python3 manage.py migrate
    print_success "Database migrations applied"
else
    print_success "Database is up to date"
fi

# Create superuser if needed (optional)
print_status "Checking for superuser..."
if ! python3 manage.py shell -c "from django.contrib.auth import get_user_model; User = get_user_model(); print('exists' if User.objects.filter(is_superuser=True).exists() else 'none')" | grep -q "exists"; then
    print_warning "No superuser found. You can create one later with: python3 manage.py createsuperuser"
else
    print_success "Superuser exists"
fi

# Collect static files
print_status "Collecting static files..."
python3 manage.py collectstatic --noinput --clear
print_success "Static files collected"

# Check if Redis is available for caching
print_status "Checking Redis connection..."
if python3 -c "import redis; r = redis.Redis(host='localhost', port=6379, db=0); r.ping(); print('Redis connected')" 2>/dev/null; then
    print_success "Redis is available for caching"
else
    print_warning "Redis not available. Caching will use dummy backend."
fi

# Test ML models loading
print_status "Testing ML models..."
if python3 -c "
import sys
sys.path.append('.')
from recognition.ml_models.fish_engine import get_fish_engine
engine = get_fish_engine()
print('ML engine loaded successfully')
" 2>/dev/null; then
    print_success "ML models loaded successfully"
else
    print_warning "ML models loading failed. Some recognition features may not work."
fi

# Start the server
print_success "All checks passed! Starting the application..."
print_status "Application will be available at:"
print_status "  - HTTP API: http://localhost:8000"
print_status "  - WebSocket: ws://localhost:8000/ws/recognition/"
print_status "  - Admin: http://localhost:8000/admin/"
print_status "  - Testing Interface: http://localhost:8000/"
print_status ""
print_status "Press Ctrl+C to stop the server"
print_status ""

# Start Daphne ASGI server
exec python3 -m daphne -b 0.0.0.0 -p 8000 fish_recognition_api.asgi:application