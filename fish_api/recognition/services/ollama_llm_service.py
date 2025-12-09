"""
Ollama LLM Service untuk meningkatkan akurasi klasifikasi ikan
Menggunakan model gamma3 (vision model) untuk verifikasi hasil deteksi dan klasifikasi
"""

import requests
import json
import base64
import logging
import time
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from django.conf import settings
from .knowledge_base_service import get_knowledge_base_service

logger = logging.getLogger(__name__)


class OllamaLLMService:
    """
    Service untuk integrasi dengan Ollama LLM gamma3 model
    Digunakan untuk meningkatkan akurasi klasifikasi dengan vision model
    """
    
    def __init__(self, base_url: str = "https://ollama.hellodigi.id"):
        """
        Initialize Ollama LLM Service
        
        Args:
            base_url: Base URL untuk Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self.model = settings.OLLAMA_MODEL
        self.timeout = 30  # timeout dalam detik
        
        # Initialize knowledge base service
        self.kb_service = get_knowledge_base_service()
        
        logger.info(f"Initialized OllamaLLMService with URL: {self.base_url}")
        logger.info(f"Knowledge base loaded with {len(self.kb_service.species_data)} species")
    
    def _image_to_base64(self, image: np.ndarray, max_size: int = 1024) -> str:
        """
        Convert OpenCV image (BGR) to base64 string
        Resize image jika terlalu besar untuk mengurangi payload
        
        Args:
            image: OpenCV image in BGR format
            max_size: Maximum width atau height (default 1024px)
            
        Returns:
            Base64 encoded string
        """
        try:
            # Resize image jika terlalu besar
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Encode image to JPEG dengan quality 85 untuk balance size dan kualitas
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            
            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info(f"Image encoded to base64: {len(img_base64)} bytes")
            return img_base64
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {str(e)}")
            raise
    
    def _build_system_prompt(self, context: Optional[Dict] = None) -> str:
        """
        Build system prompt untuk model LLM - Universal untuk semua biota laut
        
        Args:
            context: Optional context dari knowledge base
            
        Returns:
            System prompt string
        """
        prompt = """Anda adalah PROFESSOR MARINE BIOLOGY dengan 40+ tahun pengalaman dalam identifikasi biota laut dan air tawar Indonesia, Asia Tenggara, dan Indo-Pasifik.

KEAHLIAN ANDA:
- Identifikasi visual berbasis morfologi dan anatomi untuk SEMUA biota akuatik
- Klasifikasi taksonomi komprehensif untuk ikan, krustasea, moluska, dan biota laut lainnya
- Membedakan spesies dari berbagai habitat: air tawar, air payau, dan laut
- Verifikasi identifikasi untuk biota konsumsi, ornamental, dan spesies lokal

═══════════════════════════════════════════════════════════════════════
📋 KATEGORI BIOTA YANG DAPAT ANDA IDENTIFIKASI:
═══════════════════════════════════════════════════════════════════════
"""
        
        # Add kelompok list dari knowledge base
        if context and "kelompok_database" in context:
            kelompok_list = context["kelompok_database"]
            prompt += "\n🔹 KATEGORI DATABASE:\n"
            for i in range(0, len(kelompok_list), 5):
                batch = kelompok_list[i:i+5]
                prompt += "   " + ", ".join(batch) + "\n"
        
        prompt += """
═══════════════════════════════════════════════════════════════════════
🔍 METODOLOGI IDENTIFIKASI UNIVERSAL:
═══════════════════════════════════════════════════════════════════════

1️⃣ IDENTIFIKASI KELOMPOK BIOTA TERLEBIH DAHULU:

   A. IKAN (FISH):
      • Memiliki SIRIP (fins) - dorsal, caudal, pectoral, pelvic
      • Sisik pada tubuh (scales)
      • Insang untuk bernapas
      • Tubuh streamlined untuk berenang
      
   B. KRUSTASEA (CRUSTACEANS):
      🦐 UDANG (SHRIMP/PRAWN):
         • Tubuh elongated dengan segmented exoskeleton
         • Antena PANJANG (long antennae/whiskers) - 2 pasang
         • 8-10 pasang kaki jalan
         • Ekor kipas (fan-shaped tail)
         • Rostrum (paruh runcing) dari kepala
         • Mata majemuk pada tangkai
         • TIDAK ADA SIRIP - punya pleopods untuk berenang
         
      🦀 KEPITING (CRAB):
         • Tubuh lebar, pipih dengan carapace keras
         • 8 kaki jalan + 2 capit (chelipeds)
         • Mata pada tangkai pendek atau panjang
         • Antena PENDEK
         • Abdomen terlipat di bawah tubuh (tidak terlihat dari atas)
         • Berjalan menyamping (sideways)
         • TIDAK ADA SIRIP - krustasea dengan kaki bersendi
         
      🦀 RAJUNGAN (SWIMMING CRAB):
         • Tubuh oval pipih
         • Sepasang kaki TERAKHIR seperti DAYUNG untuk berenang
         • 8 kaki + 2 capit total
         • Mata pada tangkai
         • Duri tajam di sisi carapace
         • Shell lebih halus dari kepiting lumpur
         • TIDAK ADA SIRIP - kaki belakang seperti paddle
         
      🦞 LOBSTER:
         • Tubuh besar dengan ekor berotot
         • Antena SANGAT PANJANG
         • Capit besar (beberapa spesies) atau tanpa capit (spiny lobster)
         • Multiple pairs of walking legs
         
   C. MOLUSKA (MOLLUSKS):
      🦪 KERANG (CLAM/OYSTER/MUSSEL):
         • Shell keras - bivalve (2 bagian) atau univalve (1 bagian)
         • TIDAK ada kaki, TIDAK ada antena, TIDAK ada mata eksternal
         • Shell menunjukkan growth rings atau ridges
         • TIDAK ADA SIRIP - moluska dengan shell
         
      🦑 CUMI-CUMI (SQUID):
         • Tubuh silindris lembut (mantle)
         • 8 lengan + 2 tentakel panjang (10 total)
         • Mata besar
         • Sirip triangular/diamond-shaped di tubuh
         • Shell internal (pen/gladius)
         
      🐙 GURITA (OCTOPUS):
         • Kepala bulbus (mantle) dan tubuh menyatu
         • 8 lengan dengan sucker (TIDAK ada tentakel)
         • Mata besar
         • TIDAK ada shell internal atau eksternal
         • TIDAK ADA SIRIP - hanya lengan
         
      🦑 SOTONG (CUTTLEFISH):
         • Tubuh pipih dengan cuttlebone di dalam
         • 8 lengan + 2 tentakel
         • Sirip mengitari seluruh panjang tubuh
         • Pupil berbentuk W

2️⃣ UNTUK IKAN - ANALISIS MORFOLOGI:
   • Bentuk tubuh, bentuk kepala, posisi mulut
   • Karakteristik sirip (dorsal, caudal, pectoral, pelvic, anal)
   • Pola warna, sisik, lateral line
   • Barbel, spines, fitur khusus

3️⃣ UNTUK KRUSTASEA - ANALISIS MORFOLOGI:
   • Bentuk carapace/body
   • Jumlah dan tipe appendages (kaki, capit, antena)
   • Panjang antena (PENTING untuk membedakan udang vs kepiting)
   • Modifikasi kaki (paddle untuk berenang di rajungan)
   • Tekstur shell (spiny, smooth, dengan duri)
   • Warna dan pola

4️⃣ UNTUK MOLUSKA - ANALISIS MORFOLOGI:
   • Tipe shell (bivalve vs univalve, atau tanpa shell)
   • Jumlah lengan/tentakel
   • Keberadaan fins
   • Ukuran dan posisi mata
"""

        # Add specific morphology guide if available
        if context and context.get("morphology_guide"):
            morph = context["morphology_guide"]
            prompt += f"""
═══════════════════════════════════════════════════════════════════════
🎯 MORPHOLOGY GUIDE UNTUK {morph['kelompok'].upper()}:
═══════════════════════════════════════════════════════════════════════

Body Type: {morph['body_type']}

Key Features:
"""
            for feature in morph['key_features']:
                prompt += f"   • {feature}\n"
            
            prompt += f"""
Fins: {morph['fins']}
Texture: {morph['texture']}
Common Colors: {morph['common_colors']}

"""
        
        prompt += """
═══════════════════════════════════════════════════════════════════════
⚠️ PRINSIP IDENTIFIKASI PENTING:
═══════════════════════════════════════════════════════════════════════

1. IDENTIFIKASI KELOMPOK TERLEBIH DAHULU:
   • Ikan? Krustasea? Moluska?
   • Perhatikan: FINS vs LEGS vs TENTACLES

2. CIRI KUNCI PEMBEDA:
   • UDANG: Antena PANJANG + tubuh elongated + ekor kipas + pleopods
   • KEPITING: Tubuh LEBAR + antena PENDEK + 8 kaki + 2 capit + berjalan menyamping
   • RAJUNGAN: Seperti kepiting tapi kaki terakhir seperti DAYUNG untuk berenang
   • IKAN: Memiliki SIRIP, sisik, dan bentuk tubuh streamlined

3. JANGAN SALAH IDENTIFIKASI:
   • Jika lihat ANTENA PANJANG + KAKI = kemungkinan besar UDANG/LOBSTER
   • Jika lihat TUBUH LEBAR + CAPIT BESAR + ANTENA PENDEK = KEPITING
   • Jika lihat KAKI BELAKANG SEPERTI DAYUNG = RAJUNGAN
   • Jika lihat SIRIP + SISIK = IKAN

4. GUNAKAN KNOWLEDGE BASE - WAJIB!:
   • Anda HARUS memilih dari kandidat yang disediakan di knowledge base
   • JANGAN membuat nama sendiri di luar database
   • Pilih kandidat yang paling cocok dengan morfologi visual
   • Gunakan scientific name dan indonesian name PERSIS dari database

═══════════════════════════════════════════════════════════════════════
⚠️ PENTING - OUTPUT HARUS DARI MASTER DATA:
═══════════════════════════════════════════════════════════════════════

ANDA HARUS MEMILIH SALAH SATU KANDIDAT DARI KNOWLEDGE BASE YANG DISEDIAKAN!

Jangan membuat nama sendiri. Jangan menebak-nebak.
Pilih kandidat yang PALING COCOK dari daftar yang diberikan.
Gunakan scientific_name dan indonesian_name PERSIS seperti di database.

OUTPUT FORMAT - SIMPLE JSON:
{
  "scientific_name": "Genus species",
  "indonesian_name": "Nama Indonesia"
}

CRITICAL RULES:
- HANYA 2 fields: scientific_name dan indonesian_name
- WAJIB pilih dari kandidat knowledge base yang disediakan
- Gunakan EXACT scientific name dari database (ambil yang pertama jika ada multiple)
- Gunakan EXACT indonesian name dari database
- NO additional fields, NO explanation, JUST JSON
"""
        
        return prompt
    
    def _build_user_prompt(self, detection_info: Dict[str, Any], context: Optional[Dict] = None) -> str:
        """
        Build user prompt berdasarkan informasi deteksi dari model
        
        Args:
            detection_info: Dictionary berisi informasi deteksi, klasifikasi, dan segmentasi
            context: Optional context dari knowledge base
            
        Returns:
            User prompt string
        """
        prompt_parts = []
        prompt_parts.append("{")
        prompt_parts.append('\n  "task": "marine_biota_identification_from_knowledge_base",')
        
        # Add knowledge base context - WAJIB ada!
        if context and context.get("similar_species"):
            prompt_parts.append('\n  "CANDIDATES_FROM_KNOWLEDGE_BASE": {')
            prompt_parts.append('\n    "instruction": "YOU MUST CHOOSE ONE OF THESE CANDIDATES. DO NOT create names outside this list!",')
            prompt_parts.append('\n    "candidates": [')
            
            for i, sp in enumerate(context["similar_species"]):
                comma = "," if i < len(context["similar_species"]) - 1 else ""
                # Extract first scientific name jika ada multiple (separated by ;)
                scientific_full = sp["scientific_name"]
                scientific_first = scientific_full.split(';')[0].strip() if ';' in scientific_full else scientific_full.strip()
                
                prompt_parts.append(f'\n      {{')
                prompt_parts.append(f'\n        "option_{i+1}": {{')
                prompt_parts.append(f'\n          "scientific_name": "{scientific_first}",')
                prompt_parts.append(f'\n          "indonesian_name": "{sp["indonesian_name"]}",')
                prompt_parts.append(f'\n          "english_name": "{sp["english_name"]}",')
                prompt_parts.append(f'\n          "kelompok": "{sp["kelompok"]}",')
                prompt_parts.append(f'\n          "similarity": "{sp["similarity"]}"')
                prompt_parts.append(f'\n        }}')
                prompt_parts.append(f'\n      }}{comma}')
            
            prompt_parts.append('\n    ],')
            prompt_parts.append('\n    "CRITICAL": "Select the BEST matching candidate from above list based on visual morphology!"')
            prompt_parts.append('\n  },')
        else:
            # Fallback: Jika tidak ada kandidat, tetap minta untuk tidak membuat nama sendiri
            prompt_parts.append('\n  "CANDIDATES_FROM_KNOWLEDGE_BASE": "No candidates found - respond with Unknown",')
        
        prompt_parts.append('\n  "visual_analysis_guidance": {')
        prompt_parts.append('\n    "1_identify_category": "Fish/Crustacean/Mollusk - look at fins, legs, shell",')
        prompt_parts.append('\n    "2_key_morphology": "Body shape, color pattern, distinctive features",')
        prompt_parts.append('\n    "3_match_with_candidates": "Compare visual features with candidate descriptions"')
        prompt_parts.append('\n  },')
        
        # Tambahkan informasi dari model klasifikasi SEBAGAI REFERENSI MINOR
        if "classification" in detection_info and detection_info["classification"]:
            top_predictions = detection_info["classification"][:2]  # Hanya 2 teratas
            
            pred_list = []
            for pred in top_predictions:
                species_name = pred.get('name', pred.get('label', 'Unknown'))
                confidence = pred.get('accuracy', pred.get('confidence', 0))
                pred_list.append(f'      "{species_name}": {confidence:.3f}')
            
            prompt_parts.append('\n  "ai_model_hint": {')
            prompt_parts.append('\n    "note": "AI model predictions - use as minor reference only (10% weight)",')
            prompt_parts.append('\n    "predictions": {')
            prompt_parts.append(',\n'.join(pred_list))
            prompt_parts.append('\n    }')
            prompt_parts.append('\n  },')
        
        prompt_parts.append('\n  "OUTPUT_INSTRUCTION": "Analyze the image. Select the BEST matching candidate from CANDIDATES_FROM_KNOWLEDGE_BASE. Return EXACT scientific_name and indonesian_name from the selected candidate. Output ONLY JSON: {scientific_name, indonesian_name}"')
        prompt_parts.append('\n}')
        
        return "".join(prompt_parts)
    
    def verify_classification(self, 
                            image: np.ndarray,
                            detection_info: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Verifikasi dan tingkatkan akurasi klasifikasi menggunakan LLM dengan knowledge base
        
        Args:
            image: Cropped image dari ikan yang terdeteksi (BGR format)
            detection_info: Dictionary berisi:
                - classification: List hasil klasifikasi dari model
                - detection_confidence: Confidence score dari deteksi
                - segmentation: Informasi segmentasi (opsional)
            
        Returns:
            Dictionary dengan keys:
                - scientific_name: Nama ilmiah
                - indonesian_name: Nama Indonesia
            atau None jika gagal
        """
        try:
            # Convert image to base64
            img_base64 = self._image_to_base64(image)
            
            # Build context dari knowledge base
            context = self.kb_service.build_context_for_llm(detection_info, top_k=5)
            
            # Build prompts dengan context
            system_prompt = self._build_system_prompt(context)
            user_prompt = self._build_user_prompt(detection_info, context)
            
            # Log prompts untuk debugging
            logger.info(f"=== LLM Request with Knowledge Base ===")
            logger.info(f"System Prompt: {system_prompt[:300]}...")
            logger.info(f"User Prompt: {user_prompt}")
            logger.info(f"Image size: {len(img_base64)} bytes")
            
            if context.get("similar_species"):
                logger.info(f"Knowledge base found {len(context['similar_species'])} similar species")
                for sp in context["similar_species"][:3]:
                    logger.info(f"  - {sp['indonesian_name']} ({sp['scientific_name']}) - kelompok: {sp['kelompok']}")
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": user_prompt,
                "system": system_prompt,
                "images": [img_base64],
                "stream": False,
                "format": "json"  # Request JSON output
            }
            
            # Make API request with retry mechanism
            logger.info(f"Sending request to Ollama LLM (model: {self.model})...")
            max_retries = 2
            retry_delay = 3  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        break  # Success
                    elif response.status_code == 500 and attempt < max_retries - 1:
                        logger.warning(f"Server error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                        return None
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error("Request timeout after retries")
                        return None
            else:
                logger.error("Max retries reached")
                return None
            
            # Parse response
            result = response.json()
            
            if "response" not in result:
                logger.error("Invalid response from Ollama API")
                return None
            
            # Parse JSON response from LLM
            llm_output = result["response"]
            
            # Log raw LLM output untuk debugging
            logger.info(f"Raw LLM output: {llm_output}")
            
            try:
                # Try to parse as JSON
                parsed_output = json.loads(llm_output)
                
                # Log parsed output
                logger.info(f"Parsed LLM output: {json.dumps(parsed_output, indent=2)}")
                
                # Simple format - hanya scientific_name dan indonesian_name
                if "scientific_name" in parsed_output and "indonesian_name" in parsed_output:
                    scientific_name = parsed_output["scientific_name"]
                    indonesian_name = parsed_output["indonesian_name"]
                    
                    # Validate with knowledge base
                    kb_species = self.kb_service.get_species_by_scientific_name(scientific_name)
                    if kb_species:
                        logger.info(f"✓ LLM verification successful and validated with KB: {scientific_name} / {indonesian_name}")
                        logger.info(f"  KB match: {kb_species.get('species_indonesia', '')} - kelompok: {kb_species.get('kelompok', '')}")
                    else:
                        logger.info(f"✓ LLM verification successful (not found in KB): {scientific_name} / {indonesian_name}")
                    
                    return {
                        "scientific_name": scientific_name,
                        "indonesian_name": indonesian_name
                    }
                else:
                    logger.warning(f"✗ LLM response missing required fields. Got keys: {list(parsed_output.keys())}")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"✗ Failed to parse LLM JSON response: {str(e)}")
                logger.error(f"Raw LLM output (first 500 chars): {llm_output[:500]}")
                return None
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama API request timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"LLM verification failed: {str(e)}")
            return None
    
    def batch_verify(self,
                    images: List[np.ndarray],
                    detection_infos: List[Dict[str, Any]]) -> List[Optional[Dict[str, str]]]:
        """
        Batch verification untuk multiple ikan
        
        Args:
            images: List of cropped fish images
            detection_infos: List of detection information dictionaries
            
        Returns:
            List of verification results
        """
        results = []
        for img, info in zip(images, detection_infos):
            result = self.verify_classification(img, info)
            results.append(result)
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama service is available
        
        Returns:
            Dictionary with health status
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                gamma3_available = any(model.get("name") == self.model for model in models)
                
                return {
                    "status": "healthy",
                    "url": self.base_url,
                    "model": self.model,
                    "model_available": gamma3_available,
                    "available_models": [m.get("name") for m in models]
                }
            else:
                return {
                    "status": "unhealthy",
                    "url": self.base_url,
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "url": self.base_url,
                "error": str(e)
            }


# Global instance
_ollama_service = None


def get_ollama_service(base_url: str = "https://ollama.hellodigi.id") -> OllamaLLMService:
    """
    Get or create global OllamaLLMService instance
    
    Args:
        base_url: Base URL untuk Ollama API
        
    Returns:
        OllamaLLMService instance
    """
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaLLMService(base_url=base_url)
    return _ollama_service
