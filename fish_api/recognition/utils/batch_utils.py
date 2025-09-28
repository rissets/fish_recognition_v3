"""Utilities for aggregating batch recognition results."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def aggregate_species_votes(batch_results: List[Dict[str, Any]], total_images: Optional[int] = None) -> Dict[str, Any]:
    """Aggregate classification votes across batch recognition results.

    Args:
        batch_results: List of individual recognition results as returned by the
            recognition engine. Each item should contain a ``success`` flag, an
            optional ``image_index`` and a ``fish_detections`` list where each
            detection contains a ``classification`` list.
        total_images: Optional explicit total number of images processed. If not
            provided this defaults to the length of ``batch_results``.

    Returns:
        Dictionary containing aggregated statistics including per-species vote
        counts, dominant species, majority ratio and top prediction per frame.
    """

    total_images = total_images if total_images is not None else len(batch_results)

    species_votes: Dict[str, Dict[str, Any]] = {}
    per_frame_top: List[Dict[str, Any]] = []
    fish_detections_total = 0
    frames_evaluated = 0

    for result in batch_results:
        if not result or not result.get("success"):
            continue

        frames_evaluated += 1
        image_index = result.get("image_index")
        detections = result.get("fish_detections", []) or []

        best_frame_entry: Optional[Dict[str, Any]] = None

        for detection in detections:
            classification = detection.get("classification") or []
            if not classification:
                continue

            top_pred = classification[0]
            species_id = top_pred.get("species_id")
            species_name = top_pred.get("name") or "Unknown"
            vote_key = str(species_id) if species_id is not None else species_name
            detection_conf = float(detection.get("confidence", 0.0))
            accuracy = float(top_pred.get("accuracy", detection_conf))

            specie_vote = species_votes.setdefault(vote_key, {
                "species_id": species_id,
                "name": species_name,
                "count": 0,
                "best_accuracy": 0.0,
                "best_confidence": 0.0,
                "best_image_index": image_index,
            })

            specie_vote["count"] += 1
            if accuracy > specie_vote["best_accuracy"] or (
                math.isclose(accuracy, specie_vote["best_accuracy"], rel_tol=1e-6)
                and detection_conf > specie_vote["best_confidence"]
            ):
                specie_vote["best_accuracy"] = accuracy
                specie_vote["best_confidence"] = detection_conf
                specie_vote["best_image_index"] = image_index

            if (
                best_frame_entry is None
                or accuracy > best_frame_entry["accuracy"]
                or (
                    math.isclose(accuracy, best_frame_entry["accuracy"], rel_tol=1e-6)
                    and detection_conf > best_frame_entry["confidence"]
                )
            ):
                best_frame_entry = {
                    "image_index": image_index,
                    "species_name": species_name,
                    "species_id": species_id,
                    "accuracy": accuracy,
                    "confidence": detection_conf,
                }

            fish_detections_total += 1

        if best_frame_entry:
            per_frame_top.append(best_frame_entry)

    vote_list = sorted(
        species_votes.values(),
        key=lambda item: (
            -item["count"],
            -item["best_accuracy"],
            -item["best_confidence"],
        ),
    )

    top_species = vote_list[0] if vote_list else None
    total_votes = sum(item["count"] for item in vote_list)
    majority_ratio = (
        top_species["count"] / total_votes
        if top_species and total_votes > 0
        else 0.0
    )

    return {
        "frames_evaluated": frames_evaluated,
        "total_images": total_images,
        "fish_detections_total": fish_detections_total,
        "species_votes": vote_list,
        "top_species": top_species,
        "majority_ratio": majority_ratio,
        "per_frame_top": per_frame_top,
        "total_votes": total_votes,
    }

