"""
Management command to import fish master data from CSV
"""

import csv
import os
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from recognition.models import FishMasterData


class Command(BaseCommand):
    help = 'Import fish master data from CSV file'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            help='Path to CSV file (default: templates/master_data.csv)',
            default='templates/master_data.csv'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before import',
        )
    
    def handle(self, *args, **options):
        csv_file = options['file']
        clear_data = options['clear']
        
        # Resolve file path
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(settings.BASE_DIR, csv_file)
        
        if not os.path.exists(csv_file):
            raise CommandError(f'CSV file not found: {csv_file}')
        
        # Clear existing data if requested
        if clear_data:
            self.stdout.write(self.style.WARNING('Clearing existing master data...'))
            count = FishMasterData.objects.count()
            FishMasterData.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f'Deleted {count} existing records'))
        
        # Import data
        self.stdout.write(f'Importing data from: {csv_file}')
        
        created_count = 0
        updated_count = 0
        error_count = 0
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    # Get or create master data entry
                    obj, created = FishMasterData.objects.update_or_create(
                        species_indonesia=row['species_indonesia'],
                        defaults={
                            'species_english': row.get('species_english', '').strip() or None,
                            'nama_latin': row.get('nama_latin', '').strip() or '',
                            'nama_daerah': row.get('nama_daerah', '').strip() or None,
                            'kelompok': row.get('kelompok', '').strip() or None,
                            'jenis_perairan': row.get('jenis_perairan', '').strip() or None,
                            'jenis_konsumsi': row.get('jenis_konsumsi', '').strip() or None,
                            'jenis_hias': row.get('jenis_hias', '').strip() or None,
                            'jenis_dilindungi': row.get('jenis_dilindungi', '').strip() or None,
                            'prioritas': row.get('prioritas', 'HIGH').strip() or 'HIGH',
                            'search_keywords': row.get('search_keywords', '').strip() or None,
                            'min_images': int(row.get('min_images', 100)) if row.get('min_images', '').strip() else 100,
                        }
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1
                    
                    # Show progress every 100 rows
                    if (created_count + updated_count) % 100 == 0:
                        self.stdout.write(f'Processed {created_count + updated_count} rows...')
                
                except Exception as e:
                    error_count += 1
                    self.stdout.write(
                        self.style.ERROR(f'Error at row {row_num}: {str(e)}')
                    )
                    if error_count > 10:
                        raise CommandError('Too many errors, stopping import')
        
        # Summary
        self.stdout.write(self.style.SUCCESS('\n' + '='*50))
        self.stdout.write(self.style.SUCCESS('Import Summary:'))
        self.stdout.write(self.style.SUCCESS(f'  Created: {created_count}'))
        self.stdout.write(self.style.SUCCESS(f'  Updated: {updated_count}'))
        if error_count > 0:
            self.stdout.write(self.style.WARNING(f'  Errors: {error_count}'))
        self.stdout.write(self.style.SUCCESS(f'  Total: {created_count + updated_count}'))
        self.stdout.write(self.style.SUCCESS('='*50))
