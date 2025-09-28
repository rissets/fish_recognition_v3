"""Tests for the Learning-Without-Forgetting embedding adapter."""

import json
from pathlib import Path

import numpy as np
import torch

from recognition.services.lwf_adapter import LWFEmbeddingAdapter


class StubClassifier:
    def __init__(self, embeddings, labels):
        self.db_embeddings = embeddings
        self.db_labels = labels
        self.image_ids = [0]
        self.annotation_ids = [0]
        self.drawn_fish_ids = [0]
        self.keys = {1: {"label": "Existing", "species_id": 1, "embedding_count": 1}}
        self.label_to_species_id = {"Existing": 1}
        self.device = torch.device("cpu")

    def _prepare_centroids(self):
        # no-op for testing
        self.centroids_refreshed = True


class StubEngine:
    def __init__(self, classifier):
        self.classifier = classifier


def test_lwf_adapter_appends_embeddings(tmp_path):
    database_path = Path(tmp_path) / "database.pt"
    labels_path = Path(tmp_path) / "labels.json"

    initial_embeddings = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    database = {
        "embeddings": initial_embeddings.clone(),
        "labels": [1],
        "image_id": [0],
        "annotation_id": [0],
        "drawn_fish_id": [0],
        "labels_keys": {1: {"label": "Existing", "species_id": 1, "embedding_count": 1}},
    }
    torch.save(database, database_path)
    labels_path.write_text(json.dumps({"1": "Existing"}), encoding="utf-8")

    classifier = StubClassifier(
        embeddings=initial_embeddings.numpy(),
        labels=np.array([1], dtype=np.int32),
    )
    engine = StubEngine(classifier)

    def fake_embedding_fn(images):
        return torch.tensor([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]], dtype=torch.float32)

    adapter = LWFEmbeddingAdapter(
        engine=engine,
        database_path=str(database_path),
        labels_path=str(labels_path),
        embedding_fn=fake_embedding_fn,
    )

    result = adapter.adapt(
        species_name="Existing",
        images_bgr=[np.zeros((2, 2, 3), dtype=np.uint8)],
        augment=False,
    )

    assert result.species_id == 1
    assert result.new_embeddings == 2
    assert result.total_embeddings == 3
    assert np.isclose(result.majority_ratio, 2 / 3)

    saved = torch.load(database_path, map_location="cpu")
    assert saved["embeddings"].shape[0] == 3
    assert len(saved["labels"]) == 3
    assert saved["labels_keys"][1]["embedding_count"] == 3

    assert classifier.centroids_refreshed
    assert classifier.db_embeddings.shape[0] == 3
