"""
Knowledge Base Service untuk meningkatkan akurasi LLM dengan data master
Menggunakan vector similarity search untuk menemukan spesies yang relevan
"""

import csv
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """
    Service untuk mengelola knowledge base dari master_data.csv
    Menggunakan simple text matching dan fuzzy search untuk menemukan kandidat spesies
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize Knowledge Base Service
        
        Args:
            csv_path: Path ke file master_data.csv
        """
        if csv_path is None:
            csv_path = Path(settings.BASE_DIR) / "templates" / "master_data.csv"
        
        self.csv_path = csv_path
        self.species_data: List[Dict] = []
        self.kelompok_morphology: Dict[str, Dict] = {}
        
        logger.info(f"Initializing KnowledgeBaseService with CSV: {csv_path}")
        self._load_master_data()
        self._build_kelompok_morphology()
        
    def _load_master_data(self):
        """Load data dari master_data.csv"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.species_data = list(reader)
            
            logger.info(f"Loaded {len(self.species_data)} species from master data")
            
            # Log kelompok distribution
            kelompok_counts = {}
            for species in self.species_data:
                kelompok = species.get('kelompok', 'Unknown')
                kelompok_counts[kelompok] = kelompok_counts.get(kelompok, 0) + 1
            
            logger.info(f"Kelompok distribution: {kelompok_counts}")
            
        except Exception as e:
            logger.error(f"Failed to load master data: {str(e)}")
            raise
    
    def _build_kelompok_morphology(self):
        """
        Build morphological characteristics untuk setiap kelompok biota
        Ini akan digunakan untuk meningkatkan akurasi identifikasi
        """
        # Morphological characteristics untuk setiap kelompok
        self.kelompok_morphology = {
            "Udang": {
                "body_type": "crustacean",
                "key_features": [
                    "Elongated body with segmented exoskeleton",
                    "Long antennae (whiskers) - typically 2 pairs",
                    "Multiple pairs of legs (8-10 walking legs)",
                    "Fan-shaped tail (telson and uropods)",
                    "Rostrum (pointed beak) extending from head",
                    "Compound eyes on stalks",
                    "Translucent or colored hard shell"
                ],
                "fins": "NO FINS - has swimming legs (pleopods) under abdomen",
                "texture": "Hard segmented exoskeleton, jointed appendages",
                "common_colors": "translucent, white, pink, brown, orange, blue, red"
            },
            "Kepiting": {
                "body_type": "crustacean",
                "key_features": [
                    "Broad, flattened body with hard carapace",
                    "Eight walking legs + two claws (chelipeds)",
                    "Eyes on short or long stalks",
                    "Short antennae",
                    "Abdomen folded under body (not visible from top)",
                    "Sideways walking motion",
                    "Hard, thick shell (carapace)"
                ],
                "fins": "NO FINS - crustacean with jointed legs",
                "texture": "Very hard thick carapace, spines or smooth",
                "common_colors": "brown, green, red, orange, mottled patterns"
            },
            "Rajungan": {
                "body_type": "crustacean",
                "key_features": [
                    "Flattened oval body shape",
                    "Last pair of legs modified into paddles for swimming",
                    "Eight legs + two claws total",
                    "Eyes on stalks",
                    "Sharp spines on sides of carapace",
                    "Smoother shell than mud crabs",
                    "Swimming crab - paddle-like rear legs"
                ],
                "fins": "NO FINS - last leg pair are swimming paddles",
                "texture": "Hard carapace with spines, smoother than kepiting",
                "common_colors": "blue, green, brown with patterns, spots"
            },
            "Kerang": {
                "body_type": "mollusk",
                "key_features": [
                    "Hard shell - bivalve (two shell halves) or univalve (single shell)",
                    "NO legs, NO antennae, NO eyes visible externally",
                    "Shell may show growth rings or ridges",
                    "Muscle scar visible inside shells",
                    "May have siphons protruding from shell",
                    "Sessile or slow-moving"
                ],
                "fins": "NO FINS - mollusk with shell",
                "texture": "Hard calcium carbonate shell, smooth or ridged",
                "common_colors": "white, cream, brown, black, colorful patterns"
            },
            "Lobster": {
                "body_type": "crustacean",
                "key_features": [
                    "Large body with muscular tail",
                    "Very long antennae",
                    "Large claws (in some species) or no claws (spiny lobster)",
                    "Multiple pairs of walking legs",
                    "Hard, thick exoskeleton",
                    "Large size compared to shrimp"
                ],
                "fins": "NO FINS - crustacean with legs and tail fan",
                "texture": "Very hard, thick segmented exoskeleton",
                "common_colors": "brown, red, orange, green, speckled"
            },
            "Cumi-cumi": {
                "body_type": "cephalopod",
                "key_features": [
                    "Soft cylindrical body (mantle)",
                    "Eight arms + two longer tentacles (10 total)",
                    "Large eyes",
                    "Triangular or diamond-shaped fins on body",
                    "Internal shell (pen or gladius)",
                    "Can change color rapidly"
                ],
                "fins": "Two lateral fins on body (not fish fins)",
                "texture": "Soft, smooth, muscular body",
                "common_colors": "translucent white, brown, red, can change color"
            },
            "Gurita": {
                "body_type": "cephalopod",
                "key_features": [
                    "Bulbous head (mantle) and body merged",
                    "Eight arms with suckers (NO tentacles)",
                    "Large eyes",
                    "NO internal or external shell",
                    "Can change color and texture",
                    "Web of skin between arms"
                ],
                "fins": "NO FINS - cephalopod with arms only",
                "texture": "Very soft, can change texture, muscular arms",
                "common_colors": "brown, red, white, can change color dramatically"
            },
            "Sotong": {
                "body_type": "cephalopod",
                "key_features": [
                    "Flattened body with cuttlebone inside",
                    "Eight arms + two tentacles",
                    "Fins running along entire body length",
                    "W-shaped pupil",
                    "Can change color and texture",
                    "Broader, flatter body than squid"
                ],
                "fins": "Continuous undulating fin around body edge",
                "texture": "Soft body, can change texture",
                "common_colors": "brown, with zebra stripes when threatened"
            }
        }
        
        logger.info(f"Built morphology database for {len(self.kelompok_morphology)} kelompok")
    
    def get_kelompok_morphology(self, kelompok: str) -> Optional[Dict]:
        """
        Get morphological characteristics untuk kelompok tertentu
        
        Args:
            kelompok: Nama kelompok (e.g., "Udang", "Kepiting", "Rajungan")
            
        Returns:
            Dictionary berisi karakteristik morfologi atau None
        """
        return self.kelompok_morphology.get(kelompok)
    
    def search_by_kelompok(self, kelompok: str, limit: int = 10) -> List[Dict]:
        """
        Search species berdasarkan kelompok
        
        Args:
            kelompok: Nama kelompok
            limit: Maximum number of results
            
        Returns:
            List of species dictionaries
        """
        results = []
        for species in self.species_data:
            if species.get('kelompok', '').lower() == kelompok.lower():
                results.append(species)
                if len(results) >= limit:
                    break
        
        return results
    
    def find_similar_species(self, 
                            query_text: str,
                            kelompok_filter: Optional[str] = None,
                            limit: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find similar species berdasarkan text similarity
        
        Args:
            query_text: Text untuk search (scientific name, indonesian name, etc)
            kelompok_filter: Optional filter by kelompok
            limit: Maximum number of results
            
        Returns:
            List of tuples (species_dict, similarity_score)
        """
        query_lower = query_text.lower()
        results = []
        
        for species in self.species_data:
            # Apply kelompok filter
            if kelompok_filter and species.get('kelompok', '').lower() != kelompok_filter.lower():
                continue
            
            # Calculate simple text similarity score
            score = 0.0
            
            # Check species_indonesia
            if query_lower in species.get('species_indonesia', '').lower():
                score += 1.0
            
            # Check nama_latin
            nama_latin = species.get('nama_latin', '')
            if query_lower in nama_latin.lower():
                score += 1.5  # Higher weight for scientific name
            
            # Check species_english
            if query_lower in species.get('species_english', '').lower():
                score += 0.8
            
            # Check nama_daerah
            if query_lower in species.get('nama_daerah', '').lower():
                score += 0.5
            
            # Check search_keywords
            if query_lower in species.get('search_keywords', '').lower():
                score += 0.3
            
            if score > 0:
                results.append((species, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def get_kelompok_list(self) -> List[str]:
        """Get list of all kelompok in database"""
        kelompok_set = set()
        for species in self.species_data:
            kelompok = species.get('kelompok', '')
            if kelompok:
                kelompok_set.add(kelompok)
        return sorted(list(kelompok_set))
    
    def get_species_by_scientific_name(self, scientific_name: str) -> Optional[Dict]:
        """
        Get species data by scientific name
        
        Args:
            scientific_name: Scientific name to search
            
        Returns:
            Species dictionary or None
        """
        scientific_lower = scientific_name.lower()
        
        for species in self.species_data:
            nama_latin = species.get('nama_latin', '')
            # Check if scientific name is in nama_latin (may contain multiple names separated by ;)
            for latin_name in nama_latin.split(';'):
                if latin_name.strip().lower() == scientific_lower:
                    return species
        
        return None
    
    def build_context_for_llm(self, 
                             detection_info: Dict,
                             top_k: int = 5) -> Dict:
        """
        Build enhanced context untuk LLM berdasarkan detection info
        
        Args:
            detection_info: Detection information dari model
            top_k: Number of similar species to include
            
        Returns:
            Dictionary berisi enhanced context
        """
        context = {
            "kelompok_database": [],
            "similar_species": [],
            "morphology_guide": None
        }
        
        # Jika ada klasifikasi, cari similar species
        if "classification" in detection_info and detection_info["classification"]:
            top_pred = detection_info["classification"][0]
            pred_name = top_pred.get('name', top_pred.get('label', ''))
            
            # Search similar species
            similar = self.find_similar_species(pred_name, limit=top_k)
            context["similar_species"] = [
                {
                    "indonesian_name": sp.get('species_indonesia', ''),
                    "scientific_name": sp.get('nama_latin', '').split(';')[0].strip(),
                    "english_name": sp.get('species_english', ''),
                    "kelompok": sp.get('kelompok', ''),
                    "similarity": f"{score:.2f}"
                }
                for sp, score in similar
            ]
            
            # Get kelompok dari top prediction
            if similar:
                top_kelompok = similar[0][0].get('kelompok', '')
                if top_kelompok:
                    morphology = self.get_kelompok_morphology(top_kelompok)
                    if morphology:
                        context["morphology_guide"] = {
                            "kelompok": top_kelompok,
                            **morphology
                        }
        
        # Get all kelompok list
        context["kelompok_database"] = self.get_kelompok_list()
        
        return context


# Global instance
_knowledge_base_service = None


def get_knowledge_base_service(csv_path: Optional[str] = None) -> KnowledgeBaseService:
    """
    Get or create global KnowledgeBaseService instance
    
    Args:
        csv_path: Path to master_data.csv
        
    Returns:
        KnowledgeBaseService instance
    """
    global _knowledge_base_service
    if _knowledge_base_service is None:
        _knowledge_base_service = KnowledgeBaseService(csv_path=csv_path)
    return _knowledge_base_service
