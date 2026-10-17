from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

import configs.config as config
from src.detector import PersonDetector
from src.embeddings import OSNetFeatureExtractor
from src.qdrant_store import QdrantStore
from src.identity_registry import IdentityRegistry
from src.utils import crop_person_image

class ScreenshotMatcher:
    """
    Handles screenshot-based person re-identification queries:
    - Detects and crops person candidates from uploaded screenshots (or uses pre-cropped images)
    - Extracts OSNet 512-dim appearance embedding
    - Queries Qdrant vector database for candidate tracks across cameras
    - Applies validated thresholding and ambiguity checks
    - Maps retrieved tracks to registered identities in resources.json
    - Reports 'Unknown' for uncertain or low-similarity queries
    """
    def __init__(
        self,
        qstore: Optional[QdrantStore] = None,
        extractor: Optional[OSNetFeatureExtractor] = None,
        detector: Optional[PersonDetector] = None,
        registry: Optional[IdentityRegistry] = None
    ):
        self.qstore = qstore or QdrantStore()
        self.extractor = extractor or OSNetFeatureExtractor()
        self.detector = detector or PersonDetector()
        self.registry = registry or IdentityRegistry()

    def process_screenshot_crops(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects people in the uploaded screenshot.
        If people are detected, returns a list of crops and their bboxes.
        If no separate people are detected (e.g. image is already a cropped person), returns the image itself.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        # Run person detection
        detections = self.detector.detect(image)

        # If detections are found and not just a single box filling the whole frame
        candidate_crops = []
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]

            # If detection covers > 90% of image, it's essentially the whole image
            if box_w * box_h >= 0.90 * (w * h):
                continue

            crop = crop_person_image(image, bbox)
            if crop is not None:
                candidate_crops.append({
                    "crop_id": i + 1,
                    "bbox": bbox,
                    "confidence": det["confidence"],
                    "crop_image": crop
                })

        # If no specific people detected or image is already a crop (tall aspect ratio: h > w)
        if not candidate_crops:
            candidate_crops.append({
                "crop_id": 1,
                "bbox": [0, 0, w, h],
                "confidence": 1.0,
                "crop_image": image
            })

        return candidate_crops

    def _format_hit(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a raw Qdrant hit into a candidate info dict."""
        payload = hit["payload"]
        t_id = payload["track_id"]
        reg_status, assigned_id = self.registry.is_track_registered(t_id)
        return {
            "track_id": t_id,
            "camera_id": payload["camera_id"],
            "score": round(hit["score"], 4),
            "identity_label": assigned_id if reg_status else "Unregistered",
            "start_time": payload["start_time"],
            "end_time": payload["end_time"],
            "duration": payload["duration"],
            "representative_crop_path": payload["representative_crop_path"],
            "source_video_path": payload["source_video_path"]
        }

    def query_crop(
        self,
        crop_image: np.ndarray,
        top_k: int = 5,
        score_threshold: float = config.DEFAULT_MATCH_THRESHOLD,
        ambiguity_margin: float = config.AMBIGUITY_THRESHOLD,
        camera_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes query for a single person crop against Qdrant across all cameras.

        Strategy:
        1. Fetch a large pool (top_k * num_cameras) so no camera is squeezed out.
        2. Guarantee one best-match per camera (per_camera_best) for the UI panel.
        3. Merge global top-k + per-camera bests (deduplicated) for the candidates table.
        4. Use the global top-1 for the CONFIRMED/AMBIGUOUS/UNKNOWN decision.
        """
        num_cameras = len(config.CAMERA_VIDEOS)

        # 1. Extract 512-dim embedding
        query_vector = self.extractor.extract_features(crop_image)

        # 2. Fetch a large pool to guarantee per-camera coverage
        fetch_limit = max(top_k * num_cameras, 20)
        raw_hits = self.qstore.search_similar(
            query_vector=query_vector,
            limit=fetch_limit,
            camera_id=None  # Always search all cameras at this stage
        )

        # Apply optional camera filter AFTER fetching (preserves per_camera_best for full map)
        if camera_filter:
            filtered_hits = [h for h in raw_hits if h["payload"]["camera_id"] in camera_filter]
        else:
            filtered_hits = raw_hits

        if not filtered_hits:
            return {
                "decision": "UNKNOWN",
                "matched_track_id": None,
                "camera_id": None,
                "identity_label": "Unknown",
                "match_score": 0.0,
                "margin": 0.0,
                "candidates": [],
                "per_camera_best": {},
                "confirmed_camera_ids": [],
                "message": "No candidates found in Qdrant database."
            }

        # 3. Independent Per-Camera Gating & Evaluation
        # Group raw hits by camera to calculate per-camera top scores and margins
        hits_by_camera: Dict[str, List[Dict[str, Any]]] = {}
        for hit in raw_hits:
            cam = hit["payload"]["camera_id"]
            if cam not in hits_by_camera:
                hits_by_camera[cam] = []
            hits_by_camera[cam].append(hit)

        per_camera_best: Dict[str, Dict[str, Any]] = {}
        confirmed_camera_ids: List[str] = []

        for cam_id in config.CAMERA_VIDEOS.keys():
            cam_hits = hits_by_camera.get(cam_id, [])
            if not cam_hits:
                continue

            top_cam_hit = cam_hits[0]
            top_cam_score = float(top_cam_hit["score"])
            second_cam_score = float(cam_hits[1]["score"]) if len(cam_hits) > 1 else 0.0
            cam_margin = top_cam_score - second_cam_score

            # Check if top candidate and runner-up co-occur simultaneously in the same timeframe
            is_cooccurring = False
            competitor_track = None
            if len(cam_hits) > 1:
                p1 = top_cam_hit["payload"]
                p2 = cam_hits[1]["payload"]
                competitor_track = p2.get("track_id")
                start1, end1 = p1.get("start_time", 0.0), p1.get("end_time", 0.0)
                start2, end2 = p2.get("start_time", 0.0), p2.get("end_time", 0.0)
                # Overlap in time check (true physical competitors on screen at the same second)
                is_cooccurring = max(start1, start2) <= min(end1, end2)

            fmt = self._format_hit(top_cam_hit)
            fmt["margin"] = round(cam_margin, 4)
            fmt["is_cooccurring"] = is_cooccurring
            fmt["competitor_track_id"] = competitor_track
            fmt["second_score"] = round(second_cam_score, 4)

            # Independent threshold and ambiguity gating for this camera
            if top_cam_score >= score_threshold:
                # If candidates do NOT co-occur simultaneously, they are often track fragments of the same person
                # across brief occlusions. In that case, apply a relaxed margin (0.01) to avoid false rejection.
                effective_margin = ambiguity_margin if is_cooccurring else min(ambiguity_margin, 0.01)

                if cam_margin >= effective_margin:
                    fmt["status"] = "CONFIRMED_MATCH"
                    fmt["is_confirmed"] = True
                    fmt["status_label"] = "Confirmed Match"
                    confirmed_camera_ids.append(cam_id)
                else:
                    fmt["status"] = "AMBIGUOUS"
                    fmt["is_confirmed"] = False
                    fmt["status_label"] = "Ambiguous Match"
            else:
                fmt["status"] = "NOT_DETECTED"
                fmt["is_confirmed"] = False
                fmt["status_label"] = "Not Detected (Below Threshold)"

            per_camera_best[cam_id] = fmt

        # 4. Global top-k from filtered hits
        global_top = filtered_hits[:top_k]

        # 5. Analyze global top-1 for decision
        top1 = global_top[0]
        top1_score = top1["score"]
        top2_score = global_top[1]["score"] if len(global_top) > 1 else 0.0
        margin = top1_score - top2_score

        # Check registered identity for top-1
        is_reg, id_label = self.registry.is_track_registered(top1["payload"]["track_id"])
        identity_display = id_label if is_reg else "Unregistered Person"

        # 6. Apply decision logic
        if top1_score >= score_threshold:
            if margin >= ambiguity_margin:
                decision = "CONFIRMED_MATCH"
                msg = (
                    f"Confident match with track {top1['payload']['track_id']} "
                    f"({identity_display}) at similarity {top1_score:.3f} (margin: {margin:.3f})."
                )
            else:
                decision = "AMBIGUOUS"
                identity_display = "Unknown (Ambiguous Candidate)"
                msg = (
                    f"Match candidate {top1['payload']['track_id']} exceeded threshold "
                    f"({top1_score:.3f}), but top-2 margin ({margin:.3f}) is below "
                    f"ambiguity threshold ({ambiguity_margin:.2f})."
                )
        else:
            decision = "UNKNOWN"
            identity_display = "Unknown"
            msg = (
                f"Similarity score ({top1_score:.3f}) is below confidence threshold "
                f"({score_threshold:.2f}). Reported as Unknown."
            )

        # 7. Merge global top-k + per-camera bests (deduplicated, sorted by score)
        seen_track_ids = set()
        candidates_info = []
        for hit in global_top:
            fmt = self._format_hit(hit)
            if fmt["track_id"] not in seen_track_ids:
                seen_track_ids.add(fmt["track_id"])
                candidates_info.append(fmt)

        # Inject per-camera bests not already in global top-k
        for cam_id in config.CAMERA_VIDEOS.keys():
            if cam_id in per_camera_best:
                best = per_camera_best[cam_id]
                if best["track_id"] not in seen_track_ids:
                    seen_track_ids.add(best["track_id"])
                    candidates_info.append(best)

        # Sort merged list by score descending
        candidates_info.sort(key=lambda c: c["score"], reverse=True)

        return {
            "decision": decision,
            "matched_track_id": top1["payload"]["track_id"] if decision == "CONFIRMED_MATCH" else None,
            "camera_id": top1["payload"]["camera_id"] if decision == "CONFIRMED_MATCH" else None,
            "identity_label": identity_display,
            "match_score": round(top1_score, 4),
            "margin": round(margin, 4),
            "top_candidate": candidates_info[0] if candidates_info else None,
            "candidates": candidates_info,
            "per_camera_best": per_camera_best,
            "confirmed_camera_ids": confirmed_camera_ids,
            "message": msg
        }

