"""Learning-Without-Forgetting adaptation utilities for embeddings."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


logger = logging.getLogger(__name__)


@dataclass
class LWFAdaptationResult:
    species_id: int
    species_name: str
    new_embeddings: int
    total_embeddings: int
    majority_ratio: float
    centroid_shift: float
    scientific_name: Optional[str] = None


class LWFEmbeddingAdapter:
    """Perform Learning-Without-Forgetting updates on the embedding database."""

    def __init__(
        self,
        engine,
        database_path: str,
        labels_path: str,
        embedding_fn: Optional[Callable[[Sequence[np.ndarray]], torch.Tensor]] = None,
    ) -> None:
        self.engine = engine
        self.classifier = engine.classifier
        self.database_path = database_path
        self.labels_path = labels_path
        self.embedding_fn = embedding_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def adapt(
        self,
        species_name: str,
        images_bgr: Sequence[np.ndarray],
        scientific_name: Optional[str] = None,
        augment: bool = True,
    ) -> LWFAdaptationResult:
        """Adapt embeddings for a species using Learning-Without-Forgetting.

        Args:
            species_name: Target species common name.
            images_bgr: Sequence of images (OpenCV BGR format).
            scientific_name: Optional scientific name to persist.
            augment: Whether to apply light augmentation for additional samples.
        """

        if not images_bgr:
            raise ValueError("No images provided for adaptation")

        augmented_images = list(images_bgr)
        if augment:
            augmented_images.extend(self._augment_images(images_bgr))

        if self.embedding_fn is not None:
            embeddings_tensor = self.embedding_fn(augmented_images)
        else:
            embeddings_tensor = self._generate_ultra_embeddings(augmented_images)
        if embeddings_tensor.numel() == 0:
            raise ValueError("Failed to extract embeddings from provided images")

        embeddings_np = embeddings_tensor.cpu().numpy().astype(np.float32)

        database_raw = torch.load(self.database_path, map_location="cpu")
        data_store, storage_type = self._unwrap_database(database_raw)
        labels_map = self._load_labels()

        species_id = self._resolve_species_id(species_name, labels_map)

        result = self._merge_embeddings(
            species_id=species_id,
            species_name=species_name,
            scientific_name=scientific_name,
            new_embeddings=embeddings_np,
            data_store=data_store,
            labels_map=labels_map,
        )

        self._persist_database(data_store, labels_map, storage_type)
        self._refresh_classifier_state(data_store)

        try:
            self.engine._load_models()
            self.classifier = self.engine.classifier
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to reload models after LWF adaptation: {exc}")

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _unwrap_database(self, database: Any) -> (Dict[str, Any], str):
        """Normalize database structure into a mutable dictionary."""
        if isinstance(database, (list, tuple)):
            if len(database) < 5:
                raise ValueError("Unexpected embedding database structure")

            data_store = {
                'embeddings': database[0],
                'labels': list(database[1]),
                'image_id': list(database[2]),
                'annotation_id': list(database[3]),
                'drawn_fish_id': list(database[4]),
                'labels_keys': database[5] if len(database) > 5 else {},
            }
            return data_store, 'list'

        if isinstance(database, dict):
            data_store = {
                'embeddings': database.get('embeddings'),
                'labels': list(database.get('labels', [])),
                'image_id': list(database.get('image_id', [])),
                'annotation_id': list(database.get('annotation_id', [])),
                'drawn_fish_id': list(database.get('drawn_fish_id', [])),
                'labels_keys': database.get('labels_keys', {}),
            }
            return data_store, 'dict'

        raise ValueError('Unsupported embedding database format')

    def _load_labels(self) -> Dict[str, str]:
        if not os.path.exists(self.labels_path):
            return {}
        with open(self.labels_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _normalize_name(self, name: str) -> str:
        name = name.strip().lower()
        if name.startswith('ikan '):
            name = name[5:]
        return name

    def _iter_numeric_labels(self, labels_map: Dict[Any, Any]):
        for key, value in labels_map.items():
            try:
                yield int(str(key)), value
            except (TypeError, ValueError):
                continue

    def _generate_ultra_embeddings(self, images_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        embeddings: List[np.ndarray] = []
        for image in images_bgr:
            embeddings.extend(self._extract_embeddings_from_image(image))

        if not embeddings:
            dim = self.classifier.db_embeddings.shape[1] if self.classifier.db_embeddings.size else 0
            return torch.empty((0, dim), dtype=torch.float32)

        stacked = np.stack(embeddings).astype(np.float32)
        return torch.from_numpy(stacked)

    def _resolve_species_id(self, species_name: str, labels_map: Dict[str, str]) -> int:
        target_norm = self._normalize_name(species_name)

        for sid, label in self._iter_numeric_labels(labels_map):
            if self._normalize_name(label) == target_norm:
                return sid

        if hasattr(self.classifier, 'id_to_label'):
            for sid, label in self.classifier.id_to_label.items():
                try:
                    sid_int = int(str(sid))
                except (TypeError, ValueError):
                    continue
                if self._normalize_name(label) == target_norm:
                    return sid_int

        if hasattr(self.classifier, 'id_to_label'):
            for sid, label in self.classifier.id_to_label.items():
                try:
                    sid_int = int(str(sid))
                except (TypeError, ValueError):
                    continue
                normalized_label = self._normalize_name(label)
                if target_norm in normalized_label or normalized_label in target_norm:
                    return sid_int

        for sid, label in self._iter_numeric_labels(labels_map):
            normalized_label = self._normalize_name(label)
            if target_norm in normalized_label or normalized_label in target_norm:
                return sid

        numeric_ids = [sid for sid, _ in self._iter_numeric_labels(labels_map)]
        if numeric_ids:
            max_id = max(numeric_ids)
        elif len(self.classifier.db_labels):
            max_id = int(self.classifier.db_labels.max())
        else:
            max_id = 0

        new_id = max_id + 1
        labels_map[str(new_id)] = species_name
        return new_id

    def _merge_embeddings(
        self,
        *,
        species_id: int,
        species_name: str,
        scientific_name: Optional[str],
        new_embeddings: np.ndarray,
        data_store: Dict[str, Any],
        labels_map: Dict[str, str],
    ) -> LWFAdaptationResult:
        existing_embeddings = data_store.get("embeddings")
        if existing_embeddings is None:
            raise ValueError("Embedding database is missing 'embeddings' key")

        existing_labels = list(data_store.get("labels", []))
        image_ids = list(data_store.get("image_id", []))
        annotation_ids = list(data_store.get("annotation_id", []))
        drawn_ids = list(data_store.get("drawn_fish_id", []))
        labels_keys = data_store.get("labels_keys", {})

        for idx, raw_label in enumerate(existing_labels):
            try:
                existing_labels[idx] = int(raw_label)
            except (TypeError, ValueError):
                resolved_id = self._resolve_species_id(str(raw_label), labels_map)
                existing_labels[idx] = resolved_id

        current_tensor = existing_embeddings if isinstance(existing_embeddings, torch.Tensor) else torch.tensor(existing_embeddings)
        current_np = current_tensor.detach().cpu().numpy()
        new_tensor = torch.from_numpy(new_embeddings)

        existing_labels_arr = np.asarray(existing_labels, dtype=np.int64) if existing_labels else np.array([], dtype=np.int64)
        old_mask = existing_labels_arr == species_id
        old_count = int(old_mask.sum())

        base_idx = len(image_ids)
        new_count = new_embeddings.shape[0]

        updated_embeddings = torch.cat([current_tensor, new_tensor], dim=0)

        existing_labels.extend([species_id] * new_count)
        image_ids.extend(range(base_idx, base_idx + new_count))
        annotation_ids.extend(range(base_idx, base_idx + new_count))
        drawn_ids.extend(range(base_idx, base_idx + new_count))

        labels_keys = self._update_label_keys(
            labels_keys,
            species_id=species_id,
            species_name=species_name,
            scientific_name=scientific_name,
            additional_embeddings=new_count,
        )

        data_store["embeddings"] = updated_embeddings
        data_store["labels"] = existing_labels
        data_store["image_id"] = image_ids
        data_store["annotation_id"] = annotation_ids
        data_store["drawn_fish_id"] = drawn_ids
        data_store["labels_keys"] = labels_keys

        majority_ratio = new_count / (old_count + new_count) if (old_count + new_count) else 1.0

        centroid_shift = 0.0
        if old_count > 0:
            old_centroid = current_np[old_mask].mean(axis=0)
            new_centroid = new_embeddings.mean(axis=0)
            centroid_shift = float(np.linalg.norm(new_centroid - old_centroid))

        labels_map[str(species_id)] = species_name

        return LWFAdaptationResult(
            species_id=species_id,
            species_name=species_name,
            new_embeddings=new_count,
            total_embeddings=int(old_count + new_count),
            majority_ratio=float(majority_ratio),
            centroid_shift=centroid_shift,
            scientific_name=scientific_name,
        )

    def _update_label_keys(
        self,
        labels_keys: Dict[Any, Any],
        *,
        species_id: int,
        species_name: str,
        scientific_name: Optional[str],
        additional_embeddings: int,
    ) -> Dict[Any, Any]:
        labels_keys = labels_keys.copy()
        key_str = str(species_id)

        if key_str in labels_keys:
            entry = labels_keys[key_str]
            entry["embedding_count"] = entry.get("embedding_count", 0) + additional_embeddings
            entry["last_updated"] = time.time()
            if scientific_name:
                entry["scientific_name"] = scientific_name
            entry["label"] = species_name or entry.get("label")
        else:
            labels_keys[key_str] = {
                "label": species_name,
                "species_id": species_id,
                "embedding_count": additional_embeddings,
                "scientific_name": scientific_name,
                "created_at": time.time(),
            }

        return labels_keys

    def _persist_database(self, data_store: Dict[str, Any], labels_map: Dict[str, str], storage_type: str) -> None:
        if os.path.exists(self.database_path):
            backup_path = f"{self.database_path}.bak_{int(time.time())}"
            shutil.copy2(self.database_path, backup_path)

        if storage_type == 'list':
            database = [
                data_store['embeddings'],
                data_store['labels'],
                data_store['image_id'],
                data_store['annotation_id'],
                data_store['drawn_fish_id'],
                data_store['labels_keys'],
            ]
        else:
            database = {
                'embeddings': data_store['embeddings'],
                'labels': data_store['labels'],
                'image_id': data_store['image_id'],
                'annotation_id': data_store['annotation_id'],
                'drawn_fish_id': data_store['drawn_fish_id'],
                'labels_keys': data_store['labels_keys'],
            }

        torch.save(database, self.database_path)

        os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
        with open(self.labels_path, "w", encoding="utf-8") as handle:
            json.dump(labels_map, handle, ensure_ascii=False, indent=2)

    def _refresh_classifier_state(self, data_store: Dict[str, Any]) -> None:
        embeddings = data_store["embeddings"]
        labels = data_store["labels"]

        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy().astype("float32")
        else:
            embeddings_np = np.array(embeddings, dtype="float32")

        self.classifier.db_embeddings = embeddings_np
        self.classifier.db_labels = np.array(labels, dtype=np.int64)
        self.classifier.image_ids = data_store.get("image_id", [])
        self.classifier.annotation_ids = data_store.get("annotation_id", [])
        self.classifier.drawn_fish_ids = data_store.get("drawn_fish_id", [])
        self.classifier.keys = data_store.get("labels_keys", self.classifier.keys)

        self.classifier.label_to_species_id = {}
        for raw_key, value in self.classifier.keys.items():
            try:
                key_int = int(str(raw_key))
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-numeric key in labels_keys: {raw_key}")
                continue
            label = value.get("label") if isinstance(value, dict) else None
            if label:
                self.classifier.label_to_species_id[label] = value.get("species_id", key_int)

        # Provide fallback using existing map when label missing
        if not self.classifier.label_to_species_id:
            for key in np.unique(self.classifier.db_labels):
                self.classifier.label_to_species_id[str(key)] = int(key)

        if hasattr(self.classifier, 'id_to_label'):
            id_to_label = {}
            indo_map = getattr(self.classifier, 'indonesian_labels', {})
            for raw_key, value in self.classifier.keys.items():
                try:
                    key_int = int(str(raw_key))
                except (TypeError, ValueError):
                    continue
                label = value.get('label') if isinstance(value, dict) else str(value)
                species_id_lookup = value.get('species_id', key_int) if isinstance(value, dict) else key_int
                display_name = label
                if isinstance(indo_map, dict) and str(species_id_lookup) in indo_map:
                    display_name = indo_map[str(species_id_lookup)]
                id_to_label[key_int] = display_name
            self.classifier.id_to_label = id_to_label

        self.classifier._prepare_centroids()
        if hasattr(self.classifier, '_load_indonesian_labels'):
            try:
                self.classifier._load_indonesian_labels(self.database_path)
            except Exception:
                pass

    def _augment_images(self, images_bgr: Sequence[np.ndarray]) -> Iterable[np.ndarray]:
        augmented: List[np.ndarray] = []
        for image in images_bgr:
            augmented.append(self._flip_horizontal(image))
            augmented.append(self._apply_gaussian_blur(image))
            augmented.append(self._boost_saturation(image))
            augmented.append(self._enhance_contrast(image))
            augmented.append(self._add_discriminative_noise(image))
            augmented.append(self._apply_sharpening(image))
            augmented.append(self._apply_histogram_equalization(image))
            augmented.extend(self._multi_scale_crops(image))

        return [img for img in augmented if img is not None]

    def _flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    def _boost_saturation(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _add_discriminative_noise(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        noisy = image.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def _multi_scale_crops(self, image: np.ndarray) -> Iterable[np.ndarray]:
        crops: List[np.ndarray] = []
        h, w = image.shape[:2]
        scales = [0.8, 0.6]
        for scale in scales:
            nh, nw = int(h * scale), int(w * scale)
            if nh < 64 or nw < 64:
                continue
            y1 = (h - nh) // 2
            x1 = (w - nw) // 2
            crop = image[y1:y1 + nh, x1:x1 + nw]
            crops.append(cv2.resize(crop, (w, h)))
        return crops

    # ------------------------------------------------------------------
    # Ultra advanced extraction routines (adapted from ultra test harness)
    # ------------------------------------------------------------------

    def _extract_embeddings_from_image(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        embeddings: List[np.ndarray] = []
        try:
            detector = getattr(self.engine, 'detector', None)
            if detector is None:
                raise ValueError('Detector not available in engine')

            detections = detector.predict(image_bgr)[0]

            if detections:
                for detection in detections[:2]:
                    fish_crop = detection.get_mask_BGR()
                    for strategy in (
                        self._preprocess_for_embedding,
                        self._preprocess_with_attention,
                        self._preprocess_with_edge_enhancement,
                    ):
                        processed = strategy(fish_crop)
                        embedding = self._extract_single_embedding(processed)
                        if embedding is not None:
                            embeddings.append(self._apply_ultra_enhancements(embedding, processed))
            else:
                processed_image = self._preprocess_for_embedding(image_bgr)
                embedding = self._extract_single_embedding(processed_image)
                if embedding is not None:
                    embeddings.append(self._apply_ultra_enhancements(embedding, processed_image))

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to extract embeddings via ultra strategy: {exc}")

        return embeddings

    def _preprocess_for_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        return self._enhance_image(resized)

    def _preprocess_with_attention(self, image_bgr: np.ndarray) -> np.ndarray:
        processed = self._preprocess_for_embedding(image_bgr)
        attention_map = self._generate_attention_map(processed)
        return self._apply_attention_to_image(processed, attention_map)

    def _preprocess_with_edge_enhancement(self, image_bgr: np.ndarray) -> np.ndarray:
        processed = self._preprocess_for_embedding(image_bgr)
        gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        enhanced = cv2.addWeighted(processed, 0.8, edges_3ch, 0.2, 0)
        return enhanced

    def _enhance_image(self, image_rgb: np.ndarray) -> np.ndarray:
        try:
            denoised = cv2.bilateralFilter(image_rgb, 9, 75, 75)
            lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to enhance image: {exc}")
            return image_rgb

    def _generate_attention_map(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        attention = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0)
        return attention / (attention.max() + 1e-8)

    def _apply_attention_to_image(self, image_rgb: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        if attention_map.shape != image_rgb.shape[:2]:
            attention_map = cv2.resize(attention_map, (image_rgb.shape[1], image_rgb.shape[0]))
        attended = image_rgb.astype(np.float32)
        for c in range(3):
            attended[:, :, c] *= attention_map
        attended = cv2.normalize(attended, None, 0, 255, cv2.NORM_MINMAX)
        return attended.astype(np.uint8)

    def _extract_single_embedding(self, processed_rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            pil_img = Image.fromarray(processed_rgb)
            tensor = self.classifier.transform(pil_img).unsqueeze(0).to(self.classifier.device)
            with torch.no_grad():
                embeddings, _, _ = self.classifier.model(tensor)
            return embeddings.cpu().numpy().flatten()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to extract single embedding: {exc}")
            return None

    def _apply_ultra_enhancements(self, embedding: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
        normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        attention_features = self._extract_attention_features(image_rgb)
        if attention_features.size:
            if attention_features.size < normalized.size:
                padding = np.zeros(normalized.size - attention_features.size)
                attention_features = np.concatenate([attention_features, padding])
            elif attention_features.size > normalized.size:
                attention_features = attention_features[:normalized.size]
            enhanced = 0.8 * normalized + 0.2 * attention_features
            return enhanced / (np.linalg.norm(enhanced) + 1e-8)
        return normalized

    def _extract_attention_features(self, image_rgb: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            features = []
            edges = cv2.Canny(gray, 50, 150)
            features.extend([np.mean(edges), np.std(edges)])
            texture = cv2.Laplacian(gray, cv2.CV_64F)
            features.extend([np.mean(np.abs(texture)), np.std(texture)])
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            features.extend(hist.flatten())
            features = np.array(features, dtype=np.float32)
            return features / (np.linalg.norm(features) + 1e-8)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to compute attention features: {exc}")
            return np.array([])
