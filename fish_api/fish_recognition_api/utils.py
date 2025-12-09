"""
Utility functions for Django Unfold Admin
"""
from django.db.models import Count, Q, Avg
from django.utils.translation import gettext_lazy as _
from datetime import datetime, timedelta
from django.utils import timezone


def environment_callback(request):
    """
    Callback to show the environment indicator in the admin
    """
    return ["Development", "warning"]  # or "Production", "danger"


def dashboard_callback(request, context):
    """
    Callback to customize the admin dashboard with statistics and charts
    """
    from recognition.models import (
        FishIdentification,
        FishIdentificationHistory,
        FishSpeciesStatistics
    )
    
    # Calculate date ranges
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    # Overall statistics
    total_identifications = FishIdentification.objects.count()
    pending_review = FishIdentification.objects.filter(status='pending').count()
    verified_count = FishIdentification.objects.filter(status='verified').count()
    rejected_count = FishIdentification.objects.filter(status='rejected').count()
    corrected_count = FishIdentification.objects.filter(is_corrected=True).count()
    
    # AI Accuracy - based on verified identifications
    from django.db.models import F
    verified_identifications = FishIdentification.objects.filter(status='verified')
    if verified_identifications.exists():
        # Count where original matches current (AI was correct)
        correct_ai_predictions = verified_identifications.filter(
            original_scientific_name=F('current_scientific_name'),
            original_indonesian_name=F('current_indonesian_name')
        ).count()
        ai_accuracy = (correct_ai_predictions / verified_identifications.count() * 100) if verified_identifications.count() > 0 else 0
    else:
        ai_accuracy = 0
    
    # Average confidence
    avg_confidence = FishIdentification.objects.aggregate(
        avg_conf=Avg('confidence_score')
    )['avg_conf'] or 0
    
    # Recent activity (last 7 days)
    recent_identifications = FishIdentification.objects.filter(
        created_at__gte=week_ago
    ).count()
    
    recent_corrections = FishIdentification.objects.filter(
        updated_at__gte=week_ago,
        is_corrected=True
    ).count()
    
    # Top species
    top_species = FishSpeciesStatistics.objects.order_by('-total_identifications')[:5]
    
    # Species distribution for chart
    species_distribution = FishIdentification.objects.values(
        'current_scientific_name'
    ).annotate(
        count=Count('id')
    ).order_by('-count')[:10]
    
    # Daily identifications for the last 7 days
    daily_stats = []
    for i in range(6, -1, -1):
        date = today - timedelta(days=i)
        count = FishIdentification.objects.filter(
            created_at__date=date
        ).count()
        daily_stats.append({
            'date': date.strftime('%Y-%m-%d'),
            'count': count
        })
    
    # Status distribution
    status_distribution = [
        {'status': 'Pending', 'count': pending_review, 'color': '#FFA500'},
        {'status': 'Verified', 'count': verified_count, 'color': '#22C55E'},
        {'status': 'Rejected', 'count': rejected_count, 'color': '#EF4444'},
    ]
    
    context.update({
        # Key metrics
        'total_identifications': total_identifications,
        'pending_review': pending_review,
        'verified_count': verified_count,
        'rejected_count': rejected_count,
        'corrected_count': corrected_count,
        'ai_accuracy': round(ai_accuracy, 2),
        'avg_confidence': round(avg_confidence * 100, 2),
        'recent_identifications': recent_identifications,
        'recent_corrections': recent_corrections,
        
        # Charts data
        'top_species': top_species,
        'species_distribution': list(species_distribution),
        'daily_stats': daily_stats,
        'status_distribution': status_distribution,
        
        # Additional context
        'week_ago': week_ago,
        'today': today,
    })
    
    return context
