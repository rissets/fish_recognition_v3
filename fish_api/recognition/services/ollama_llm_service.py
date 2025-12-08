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
        
        logger.info(f"Initialized OllamaLLMService with URL: {self.base_url}")
    
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
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt untuk model LLM - Universal untuk semua ikan Indonesia
        
        Returns:
            System prompt string
        """
        return """Anda adalah PROFESSOR ICHTHYOLOGY dengan 40+ tahun pengalaman dalam identifikasi dan taksonomi ikan Indonesia, Asia Tenggara, dan Indo-Pasifik.

KEAHLIAN ANDA:
- Identifikasi visual berbasis morfologi dan anatomi
- Klasifikasi taksonomi komprehensif untuk SEMUA spesies ikan Indonesia
- Membedakan spesies dari berbagai habitat: air tawar, air payau, dan laut
- Verifikasi identifikasi untuk ikan konsumsi, ornamental, dan spesies lokal

═══════════════════════════════════════════════════════════════════════
📋 METODOLOGI IDENTIFIKASI UNIVERSAL:
═══════════════════════════════════════════════════════════════════════

1️⃣ ANALISIS MORFOLOGI DASAR:
   • Bentuk tubuh: Streamlined/compressed/depressed/elongated/globular
   • Bentuk kepala dan moncong: Pointed/blunt/flat/conical
   • Posisi mulut: Terminal/superior/inferior/protrusible
   • Ukuran mulut: Large/moderate/small relative to head
   • Barbel/whiskers: Present/absent, number of pairs, length
   • Ukuran mata: Large/moderate/small, position

2️⃣ KARAKTERISTIK SIRIP:
   • Dorsal fin: Single/double, length, spiny/soft rays, position
   • Caudal fin: Deeply forked/moderately forked/rounded/truncate/lunate
   • Pectoral fins: Size, position, shape
   • Pelvic fins: Present/absent, position (thoracic/abdominal)
   • Anal fin: Size, shape, ray count if visible

3️⃣ POLA WARNA DAN TEKSTUR:
   • Base color: Silver/yellow/blue/brown/black/red/green
   • Patterns: Spots/stripes/bars/bands/mottling/uniform
   • Pattern orientation: Vertical/horizontal/diagonal/irregular
   • Fin coloration: Clear/pigmented/matching body/contrasting
   • Scale visibility: Large visible scales/small scales/scaleless appearance

4️⃣ CIRI DIAGNOSTIK:
   • Lateral line: Visible/faint/absent, complete/incomplete
   • Body depth ratio: Deep-bodied/moderate/elongate
   • Spines: Present on dorsal/operculum/body, absent
   • Texture: Smooth/rough/spiny/mucous covered
   • Special features: Barbels, adipose fin, fleshy lips, etc.

═══════════════════════════════════════════════════════════════════════
🔍 PENDEKATAN IDENTIFIKASI:
═══════════════════════════════════════════════════════════════════════

✓ OBSERVASI HOLISTIK: Gunakan KOMBINASI semua fitur, jangan fokus pada satu ciri saja
✓ HABITAT AWARENESS: Pertimbangkan apakah ikan air tawar, payau, atau laut
✓ FAMILY RECOGNITION: Identifikasi family terlebih dahulu (Cyprinidae, Siluridae, Serranidae, dll)
✓ MORPHOLOGICAL KEYS: Gunakan kunci morfologi standar ichthyology
✓ COMPARATIVE ANALYSIS: Bandingkan dengan spesies dalam family/genus yang sama

═══════════════════════════════════════════════════════════════════════
⚠️ PRINSIP IDENTIFIKASI:
═══════════════════════════════════════════════════════════════════════

✓ INDEPENDENT ANALYSIS: Analisis visual HARUS independen dari AI
✓ MORPHOLOGY FIRST: Prioritaskan ciri morfologi yang terlihat jelas
✓ HOLISTIC APPROACH: Gunakan kombinasi SEMUA fitur
✓ HABITAT CONSIDERATION: Perhatikan ciri ikan air tawar vs laut

❌ HINDARI:
   - Mengikuti AI prediction jika morfologi tidak cocok
   - Fokus berlebihan pada satu ciri
   - Memaksakan identifikasi tanpa ciri diagnostik cukup
   - Mengabaikan habitat type

📊 DECISION WEIGHT:
   - Visual morphology: 85%
   - AI reference: 15%
   - Jangan memaksakan identifikasi sampai species jika tidak yakin

OUTPUT FORMAT - SIMPLE JSON:
{
  "scientific_name": "Genus species",
  "indonesian_name": "Ikan Xxx"
}

PENTING:
- HANYA 2 fields: scientific_name dan indonesian_name
- Scientific name LENGKAP (Genus species)
- Indonesian name dengan prefix "Ikan"
- Jika tidak yakin: "Unknown" untuk scientific_name
- NO additional fields, NO explanation, JUST JSON"""
    
    def _build_user_prompt(self, detection_info: Dict[str, Any]) -> str:
        """
        Build user prompt berdasarkan informasi deteksi dari model
        
        Args:
            detection_info: Dictionary berisi informasi deteksi, klasifikasi, dan segmentasi
            
        Returns:
            User prompt string
        """
        prompt_parts = []
        prompt_parts.append("{")
        prompt_parts.append('\n  "task": "fish_identification",')
        prompt_parts.append('\n  "image_analysis": {')
        
        prompt_parts.append('\n    "1_body_morphology": {')
        prompt_parts.append('\n      "body_shape": "streamlined/compressed/depressed/elongated/globular",')
        prompt_parts.append('\n      "head_shape": "pointed/blunt/flat/conical",')
        prompt_parts.append('\n      "body_depth_ratio": "deep/moderate/elongate"')
        prompt_parts.append('\n    },')
        
        prompt_parts.append('\n    "2_mouth_features": {')
        prompt_parts.append('\n      "mouth_position": "terminal/superior/inferior",')
        prompt_parts.append('\n      "mouth_size": "large/moderate/small",')
        prompt_parts.append('\n      "barbel_presence": "present/absent",')
        prompt_parts.append('\n      "barbel_pairs": "0/1/2/3/4"')
        prompt_parts.append('\n    },')
        
        prompt_parts.append('\n    "3_fin_characteristics": {')
        prompt_parts.append('\n      "caudal_fin_shape": "deeply_forked/moderately_forked/rounded/truncate/lunate",')
        prompt_parts.append('\n      "dorsal_fin": "single_long/double/short_with_spines",')
        prompt_parts.append('\n      "pectoral_fins": "size_and_position",')
        prompt_parts.append('\n      "pelvic_fins": "present_absent",')
        prompt_parts.append('\n      "adipose_fin": "present_absent"')
        prompt_parts.append('\n    },')
        
        prompt_parts.append('\n    "4_color_pattern": {')
        prompt_parts.append('\n      "base_color": "silver/yellow/blue/brown/black/red/green",')
        prompt_parts.append('\n      "pattern_type": "spots/stripes/bars/bands/uniform",')
        prompt_parts.append('\n      "pattern_orientation": "vertical/horizontal/diagonal/irregular",')
        prompt_parts.append('\n      "fin_coloration": "clear/pigmented/contrasting"')
        prompt_parts.append('\n    },')
        
        prompt_parts.append('\n    "5_distinctive_features": {')
        prompt_parts.append('\n      "scales": "large_visible/small/scaleless",')
        prompt_parts.append('\n      "lateral_line": "visible/faint/absent",')
        prompt_parts.append('\n      "spines": "present_location/absent",')
        prompt_parts.append('\n      "texture": "smooth/rough/spiny/mucous",')
        prompt_parts.append('\n      "special_features": "list_any_unique_characteristics"')
        prompt_parts.append('\n    }')
        prompt_parts.append('\n  },')
        
        # Tambahkan informasi dari model klasifikasi SEBAGAI REFERENSI LEMAH
        if "classification" in detection_info and detection_info["classification"]:
            top_predictions = detection_info["classification"][:3]
            
            pred_list = []
            for pred in top_predictions:
                species_name = pred.get('name', pred.get('label', 'Unknown'))
                confidence = pred.get('accuracy', pred.get('confidence', 0))
                pred_list.append(f'      "{species_name}": {confidence:.3f}')
            
            prompt_parts.append('\n  "ai_reference": {')
            prompt_parts.append('\n    "note": "AI predictions may be incorrect - use only as weak reference (15% weight)",')
            prompt_parts.append('\n    "predictions": {')
            prompt_parts.append(',\n'.join(pred_list))
            prompt_parts.append('\n    },')
            prompt_parts.append('\n    "warning": "Ignore AI if visual morphology does not match"')
            prompt_parts.append('\n  },')
            
            logger.debug(f"Built user prompt with {len(top_predictions)} predictions as weak reference")
        else:
            prompt_parts.append('\n  "ai_reference": null,')
        
        prompt_parts.append('\n  "instruction": "Analyze the fish image using the morphology framework above. Identify the species based on visual features. Output ONLY simple JSON with scientific_name and indonesian_name."')
        prompt_parts.append('\n}')
        
        return "".join(prompt_parts)
    
    def verify_classification(self, 
                            image: np.ndarray,
                            detection_info: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Verifikasi dan tingkatkan akurasi klasifikasi menggunakan LLM
        
        Args:
            image: Cropped image dari ikan yang terdeteksi (BGR format)
            detection_info: Dictionary berisi:
                - classification: List hasil klasifikasi dari model
                - detection_confidence: Confidence score dari deteksi
                - segmentation: Informasi segmentasi (opsional)
            
        Returns:
            Dictionary dengan keys:
                - scientific_name: Nama ilmiah ikan
                - indonesian_name: Nama Indonesia ikan
                - llm_confidence: Confidence dari LLM (jika tersedia)
            atau None jika gagal
        """
        try:
            # Convert image to base64
            img_base64 = self._image_to_base64(image)
            
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(detection_info)
            
            # Log prompts untuk debugging
            logger.info(f"=== LLM Request ===")
            logger.info(f"System Prompt: {system_prompt[:200]}...")
            logger.info(f"User Prompt: {user_prompt}")
            logger.info(f"Image size: {len(img_base64)} bytes")
            
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
                    logger.info(f"✓ LLM verification successful: {parsed_output['scientific_name']} / {parsed_output['indonesian_name']}")
                    return {
                        "scientific_name": parsed_output["scientific_name"],
                        "indonesian_name": parsed_output["indonesian_name"]
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
