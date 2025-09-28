"""
Simple view to serve the testing app
"""

from django.shortcuts import render


def index(request):
    """Serve the main testing application"""
    return render(request, 'index.html')