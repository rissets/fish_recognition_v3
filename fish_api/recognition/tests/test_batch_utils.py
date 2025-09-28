"""Unit tests for batch aggregation utilities."""

import math

from recognition.utils.batch_utils import aggregate_species_votes


def build_detection(species_name, species_id, accuracy, confidence):
    return {
        "confidence": confidence,
        "classification": [
            {
                "name": species_name,
                "species_id": species_id,
                "accuracy": accuracy,
            }
        ],
    }


def test_aggregate_species_votes_basic():
    batch_results = [
        {
            "success": True,
            "image_index": 0,
            "fish_detections": [
                build_detection("Ikan Mujair", 12, 0.9, 0.88),
                build_detection("Ikan Mujair", 12, 0.86, 0.80),
            ],
        },
        {
            "success": True,
            "image_index": 1,
            "fish_detections": [
                build_detection("Ikan Mujair", 12, 0.94, 0.91),
                build_detection("Ikan Nila", 13, 0.83, 0.82),
            ],
        },
    ]

    summary = aggregate_species_votes(batch_results, total_images=2)

    assert summary["frames_evaluated"] == 2
    assert summary["fish_detections_total"] == 4
    assert summary["top_species"]["name"] == "Ikan Mujair"
    assert summary["top_species"]["count"] == 3
    assert math.isclose(summary["majority_ratio"], 0.75)

    per_frame = summary["per_frame_top"]
    assert len(per_frame) == 2
    assert per_frame[0]["image_index"] == 0
    assert per_frame[1]["species_id"] == 12
