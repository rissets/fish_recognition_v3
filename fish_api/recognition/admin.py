"""
Django Unfold Admin Configuration for Fish Recognition System
"""
from django.contrib import admin
from django.db.models import Count, Avg, Q, F
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from django.urls import reverse
from django.utils import timezone
from unfold.admin import ModelAdmin, TabularInline
from unfold.decorators import display
from unfold.contrib.filters.admin import (
    RangeDateFilter,
    RangeDateTimeFilter,
    RelatedDropdownFilter,
    DropdownFilter,
)
from .models import (
    FishIdentification, 
    FishIdentificationHistory, 
    FishSpeciesStatistics
)


class FishIdentificationHistoryInline(TabularInline):
    """Inline admin for showing identification history"""
    model = FishIdentificationHistory
    extra = 0
    readonly_fields = [
        'changed_at', 'changed_by', 'field_name', 
        'old_value', 'new_value', 'change_reason'
    ]
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False


class FishIdentificationAdmin(ModelAdmin):
    """Admin interface for Fish Identifications with enhanced features"""
    
    # Unfold icon for sidebar
    icon = "visibility"
    
    list_display = [
        'display_thumbnail',
        'current_scientific_name',
        'current_english_name',
        'display_confidence',
        'status_badge',
        'is_corrected_badge',
        'created_at',
        'actions_column',
    ]
    
    list_filter = [
        'status',
        'is_corrected',
        ('created_at', RangeDateTimeFilter),
        ('updated_at', RangeDateTimeFilter),
        'current_scientific_name',
        'original_scientific_name',
    ]
    
    search_fields = [
        'current_scientific_name',
        'current_english_name',
        'current_indonesian_name',
        'original_scientific_name',
        'original_english_name',
        'id',
    ]
    
    readonly_fields = [
        'id',
        'created_at',
        'updated_at',
        'display_original_image',
        'display_thumbnail_preview',
        'original_scientific_name',
        'original_english_name',
        'original_indonesian_name',
        'ai_model_version',
        'kb_candidates',
    ]
    
    fieldsets = (
        (_('Identification Information'), {
            'fields': (
                'id',
                'status',
                'is_corrected',
                'user_identifier',
            ),
        }),
        (_('Current Classification'), {
            'fields': (
                'current_scientific_name',
                'current_english_name',
                'current_indonesian_name',
                'current_kelompok',
                'confidence_score',
            ),
            'description': _('Current fish classification (can be corrected by users)'),
        }),
        (_('Original AI Classification'), {
            'fields': (
                'original_scientific_name',
                'original_english_name',
                'original_indonesian_name',
                'original_kelompok',
                'ai_model_version',
                'kb_candidates',
            ),
            'classes': ('collapse',),
            'description': _('Original classification by AI model (immutable)'),
        }),
        (_('Images'), {
            'fields': (
                'display_original_image',
                'display_thumbnail_preview',
                'image_url',
                'thumbnail_url',
            ),
        }),
        (_('Detection & Location'), {
            'fields': (
                'detection_box',
                'detection_score',
                'user_location',
            ),
            'classes': ('collapse',),
        }),
        (_('Correction Info'), {
            'fields': (
                'corrected_at',
                'correction_notes',
            ),
            'classes': ('collapse',),
        }),
        (_('Metadata'), {
            'fields': (
                'created_at',
                'updated_at',
            ),
            'classes': ('collapse',),
        }),
    )
    
    inlines = [FishIdentificationHistoryInline]
    
    actions = ['verify_identifications', 'mark_as_pending']
    
    @display(description=_("Thumbnail"), ordering="current_scientific_name")
    def display_thumbnail(self, obj):
        """Display thumbnail image in list view"""
        # Prioritize MinIO URL over local file
        if obj.thumbnail_url:
            return format_html(
                '<img src="{}" style="width: 60px; height: 60px; object-fit: cover; border-radius: 8px;" />',
                obj.thumbnail_url
            )
        elif obj.thumbnail:
            return format_html(
                '<img src="{}" style="width: 60px; height: 60px; object-fit: cover; border-radius: 8px;" />',
                obj.thumbnail.url
            )
        return format_html('<span style="color: #999;">No image</span>')
    
    @display(description=_("Confidence"), ordering="confidence_score")
    def display_confidence(self, obj):
        """Display confidence score with color coding"""
        confidence = obj.confidence_score * 100
        if confidence >= 80:
            color = '#22C55E'  # Green
        elif confidence >= 60:
            color = '#FFA500'  # Orange
        else:
            color = '#EF4444'  # Red
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}%</span>',
            color,
            round(confidence, 1)
        )
    
    @display(description=_("Status"), ordering="status")
    def status_badge(self, obj):
        """Display status as colored badge"""
        colors = {
            'pending': '#FFA500',
            'verified': '#22C55E',
            'rejected': '#EF4444',
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 500;">{}</span>',
            colors.get(obj.status, '#999'),
            obj.get_status_display()
        )
    
    @display(description=_("Corrected"), ordering="is_corrected", boolean=True)
    def is_corrected_badge(self, obj):
        """Display if identification was corrected"""
        return obj.is_corrected
    
    @display(description=_("Actions"))
    def actions_column(self, obj):
        """Display action buttons"""
        view_url = reverse('admin:recognition_fishidentification_change', args=[obj.id])
        return format_html(
            '<a href="{}" style="color: #3B82F6; text-decoration: none;">View Details</a>',
            view_url
        )
    
    @display(description=_("Original Image"))
    def display_original_image(self, obj):
        """Display original image in detail view"""
        if obj.image_url:
            return format_html(
                '<img src="{}" style="max-width: 400px; max-height: 400px; border-radius: 8px;" />',
                obj.image_url
            )
        elif obj.image:
            return format_html(
                '<img src="{}" style="max-width: 400px; max-height: 400px; border-radius: 8px;" />',
                obj.image.url
            )
        return format_html('<span style="color: #999;">No image</span>')
    
    @display(description=_("Thumbnail Preview"))
    def display_thumbnail_preview(self, obj):
        """Display thumbnail in detail view"""
        if obj.image_url:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px; border-radius: 8px;" />',
                obj.image_url
            )
        elif obj.thumbnail:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px; border-radius: 8px;" />',
                obj.thumbnail.url
            )
        return format_html('<span style="color: #999;">No thumbnail</span>')
    
    @admin.action(description=_("Verify selected identifications"))
    def verify_identifications(self, request, queryset):
        """Bulk action to verify identifications"""
        updated = queryset.filter(status='pending').update(
            status='verified',
            updated_at=timezone.now()
        )
        self.message_user(
            request,
            _('Successfully verified {} identification(s).').format(updated),
        )
    
    @admin.action(description=_("Mark as pending review"))
    def mark_as_pending(self, request, queryset):
        """Bulk action to mark identifications as pending"""
        updated = queryset.update(
            status='pending',
            updated_at=timezone.now()
        )
        self.message_user(
            request,
            _('Successfully marked {} identification(s) as pending.').format(updated),
        )
    
    def get_queryset(self, request):
        """Optimize queryset with select_related"""
        qs = super().get_queryset(request)
        return qs.select_related()
    
    class Meta:
        verbose_name = _("Fish Identification")
        verbose_name_plural = _("Fish Identifications")


class FishIdentificationHistoryAdmin(ModelAdmin):
    """Admin interface for Fish Identification History"""
    
    # Unfold icon for sidebar
    icon = "history"
    
    list_display = [
        'identification',
        'field_name',
        'display_change',
        'changed_by',
        'changed_at',
    ]
    
    list_filter = [
        'field_name',
        ('changed_at', RangeDateTimeFilter),
        'changed_by',
    ]
    
    search_fields = [
        'identification__current_scientific_name',
        'identification__id',
        'field_name',
        'old_value',
        'new_value',
        'change_reason',
    ]
    
    readonly_fields = [
        'identification',
        'changed_at',
        'changed_by',
        'field_name',
        'old_value',
        'new_value',
        'change_reason',
    ]
    
    fieldsets = (
        (_('Change Information'), {
            'fields': (
                'identification',
                'field_name',
                'changed_by',
                'changed_at',
                'change_reason',
            ),
        }),
        (_('Value Changes'), {
            'fields': (
                'old_value',
                'new_value',
            ),
        }),
    )
    
    @display(description=_("Change"))
    def display_change(self, obj):
        """Display summary of changes"""
        return format_html(
            '<div><strong>{}:</strong> {} → {}</div>',
            obj.field_name,
            obj.old_value or 'N/A',
            obj.new_value or 'N/A'
        )
    
    def has_add_permission(self, request):
        """Disable manual addition of history records"""
        return False
    
    def has_delete_permission(self, request, obj=None):
        """Disable deletion of history records"""
        return False
    
    class Meta:
        verbose_name = _("Identification History")
        verbose_name_plural = _("Identification Histories")


class FishSpeciesStatisticsAdmin(ModelAdmin):
    """Admin interface for Fish Species Statistics"""
    
    # Unfold icon for sidebar
    icon = "analytics"
    
    list_display = [
        'scientific_name',
        'indonesian_name',
        'display_total_identifications',
        'display_accuracy',
        'display_avg_confidence',
        'last_seen',
    ]
    
    list_filter = [
        'scientific_name',
        'kelompok',
        ('last_seen', RangeDateTimeFilter),
    ]
    
    search_fields = [
        'scientific_name',
        'indonesian_name',
        'english_name',
        'kelompok',
    ]
    
    readonly_fields = [
        'scientific_name',
        'indonesian_name',
        'english_name',
        'kelompok',
        'total_identifications',
        'correct_identifications',
        'corrected_identifications',
        'average_confidence',
        'last_seen',
        'first_seen',
    ]
    
    fieldsets = (
        (_('Species Information'), {
            'fields': (
                'scientific_name',
                'indonesian_name',
                'english_name',
                'kelompok',
            ),
        }),
        (_('Statistics'), {
            'fields': (
                'total_identifications',
                'correct_identifications',
                'corrected_identifications',
                'average_confidence',
            ),
        }),
        (_('Timeline'), {
            'fields': (
                'first_seen',
                'last_seen',
            ),
        }),
    )
    
    @display(description=_("Total"), ordering="total_identifications")
    def display_total_identifications(self, obj):
        """Display total identifications with formatting"""
        return format_html(
            '<span style="font-weight: bold; color: #3B82F6;">{}</span>',
            obj.total_identifications
        )
    
    @display(description=_("Accuracy"), ordering="correct_identifications")
    def display_accuracy(self, obj):
        """Display accuracy percentage with color coding"""
        if obj.total_identifications > 0:
            accuracy = (obj.correct_identifications / obj.total_identifications) * 100
        else:
            accuracy = 0
        
        if accuracy >= 80:
            color = '#22C55E'  # Green
        elif accuracy >= 60:
            color = '#FFA500'  # Orange
        else:
            color = '#EF4444'  # Red
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}%</span>',
            color,
            round(accuracy, 1)
        )
    
    @display(description=_("Avg Confidence"), ordering="average_confidence")
    def display_avg_confidence(self, obj):
        """Display average confidence with formatting"""
        confidence = (obj.average_confidence or 0) * 100
        return format_html(
            '<span style="font-weight: 500;">{}%</span>',
            round(confidence, 1)
        )
    
    def has_add_permission(self, request):
        """Disable manual addition of statistics"""
        return False
    
    def has_delete_permission(self, request, obj=None):
        """Disable deletion of statistics"""
        return False
    
    class Meta:
        verbose_name = _("Species Statistics")
        verbose_name_plural = _("Species Statistics")


class FishMasterDataAdmin(ModelAdmin):
    """Admin interface for Fish Master Data"""
    
    # Unfold icon for sidebar
    icon = "dataset"
    
    list_display = [
        'species_indonesia',
        'species_english',
        'kelompok',
        'display_jenis_perairan',
        'display_konsumsi_badge',
        'display_hias_badge',
        'display_dilindungi_badge',
        'prioritas',
        'min_images',
        'updated_at',
    ]
    
    list_filter = [
        'kelompok',
        'jenis_perairan',
        'jenis_konsumsi',
        'jenis_hias',
        'jenis_dilindungi',
        'prioritas',
        ('updated_at', RangeDateTimeFilter),
    ]
    
    search_fields = [
        'species_indonesia',
        'species_english',
        'nama_latin',
        'nama_daerah',
        'kelompok',
        'search_keywords',
    ]
    
    readonly_fields = [
        'created_at',
        'updated_at',
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': (
                'species_indonesia',
                'species_english',
                'nama_latin',
                'nama_daerah',
                'kelompok',
            ),
        }),
        (_('Classification'), {
            'fields': (
                'jenis_perairan',
                'jenis_konsumsi',
                'jenis_hias',
                'jenis_dilindungi',
                'prioritas',
            ),
        }),
        (_('Metadata'), {
            'fields': (
                'search_keywords',
                'min_images',
            ),
        }),
        (_('Timestamps'), {
            'fields': (
                'created_at',
                'updated_at',
            ),
        }),
    )
    
    list_per_page = 50
    
    @display(description=_("Jenis Perairan"))
    def display_jenis_perairan(self, obj):
        """Display jenis perairan with badges"""
        if not obj.jenis_perairan:
            return '-'
        
        colors = {
            'LAUT': '#3B82F6',
            'TAWAR': '#10B981',
            'PAYAU': '#F59E0B',
        }
        
        badges = []
        for jenis in obj.jenis_perairan.split(','):
            jenis = jenis.strip()
            color = colors.get(jenis, '#6B7280')
            badges.append(
                f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
                f'border-radius: 4px; font-size: 11px; margin-right: 4px;">{jenis}</span>'
            )
        
        return format_html(''.join(badges))
    
    @display(description=_("Konsumsi"), boolean=True)
    def display_konsumsi_badge(self, obj):
        """Display consumption status"""
        return obj.jenis_konsumsi == 'KONSUMSI'
    
    @display(description=_("Hias"), boolean=True)
    def display_hias_badge(self, obj):
        """Display ornamental status"""
        return obj.jenis_hias == 'HIAS'
    
    @display(description=_("Dilindungi"), boolean=True)
    def display_dilindungi_badge(self, obj):
        """Display protected status"""
        return obj.jenis_dilindungi == 'YA'
    
    actions = ['export_to_csv']
    
    @admin.action(description=_("Export selected to CSV"))
    def export_to_csv(self, request, queryset):
        """Export selected master data to CSV"""
        import csv
        from django.http import HttpResponse
        from datetime import datetime
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="master_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'species_indonesia', 'species_english', 'nama_latin', 'nama_daerah',
            'kelompok', 'jenis_perairan', 'jenis_konsumsi', 'jenis_hias',
            'jenis_dilindungi', 'prioritas', 'search_keywords', 'min_images'
        ])
        
        for obj in queryset:
            writer.writerow([
                obj.species_indonesia,
                obj.species_english or '',
                obj.nama_latin or '',
                obj.nama_daerah or '',
                obj.kelompok or '',
                obj.jenis_perairan or '',
                obj.jenis_konsumsi or '',
                obj.jenis_hias or '',
                obj.jenis_dilindungi or '',
                obj.prioritas,
                obj.search_keywords or '',
                obj.min_images,
            ])
        
        return response


# Register models with admin site
from .models import FishIdentification, FishIdentificationHistory, FishSpeciesStatistics, FishMasterData

admin.site.register(FishIdentification, FishIdentificationAdmin)
admin.site.register(FishIdentificationHistory, FishIdentificationHistoryAdmin)
admin.site.register(FishSpeciesStatistics, FishSpeciesStatisticsAdmin)
admin.site.register(FishMasterData, FishMasterDataAdmin)

