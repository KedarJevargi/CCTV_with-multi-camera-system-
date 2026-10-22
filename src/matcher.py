from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

import configs.config as config
from src.detector import PersonDetector
from src.embeddings import OSNetFeatureExtractor
from src.qdrant_store import QdrantStore
from src.identity_registry import IdentityRegistry
from src.face_module import FaceBiometricModule
from src.utils import crop_person_image

class ScreenshotMatcher:
    """
    Handles screenshot-based person re-identification queries:
    - Detects and crops person candidates from uploaded screenshots (or uses pre-cropped images)
    - Stage 1: Extracts OSNet 512-dim appearance embedding for fast vector search in Qdrant
    - Stage 2: Biometric Face Verification (YuNet + SFace) to confirm matches and veto bystanders in similar clothing
    - Applies validated thresholding and ambiguity checks
    - Maps retrieved tracks to registered identities in resources.json
    - Reports 'Unknown' for uncertain or low-similarity queries
    """
    def __init__(
        self,
        qstore: Optional[QdrantStore] = None,
        extractor: Optional[OSNetFeatureExtractor] = None,
        detector: Optional[PersonDetector] = None,
        registry: Optional[IdentityRegistry] = None,
        face_module: Optional[FaceBiometricModule] = None
    ):
        self.qstore = qstore or QdrantStore()
        self.extractor = extractor or OSNetFeatureExtractor()
        self.detector = detector or PersonDetector()
        self.registry = registry or IdentityRegistry()
        self.face_module = face_module or FaceBiometricModule()

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
        camera_filter: Optional[List[str]] = None,
        face_mode: str = "Adaptive"
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

        # 1. Stage 1: Extract OSNet 512-dim appearance embedding
        query_vector = self.extractor.extract_features(crop_image)

        # Stage 2: Extract Face Biometric embedding if face detected in query and face_mode is active
        if face_mode != "Appearance Only":
            query_face_info = self.face_module.detect_face(crop_image)
            query_face_emb = self.face_module.extract_embedding(crop_image, query_face_info) if query_face_info else None
        else:
            query_face_info = None
            query_face_emb = None

        def verify_crop_face(crop_path: Optional[str]) -> Dict[str, Any]:
            if not crop_path or not Path(crop_path).exists() or query_face_emb is None:
                return {
                    "face_status": "UNAVAILABLE",
                    "face_similarity": None,
                    "face_verdict": "UNAVAILABLE",
                    "face_message": "Face not available for verification"
                }
            cand_img = cv2.imread(crop_path)
            if cand_img is None:
                return {
                    "face_status": "UNAVAILABLE",
                    "face_similarity": None,
                    "face_verdict": "UNAVAILABLE",
                    "face_message": "Candidate crop image unreadable"
                }
            cand_face_emb = self.face_module.extract_embedding(cand_img)
            ver = self.face_module.verify_pair(query_face_emb, cand_face_emb)
            return {
                "face_status": ver["verdict"],
                "face_similarity": ver["similarity"],
                "face_verdict": ver["verdict"],
                "face_message": ver["message"]
            }

        # 2. Fetch candidate pools for each camera independently to guarantee 100% per-camera recall
        target_cameras = [c for c in (camera_filter if camera_filter else config.CAMERA_VIDEOS.keys())]
        hits_by_camera: Dict[str, List[Dict[str, Any]]] = {}
        all_hits: List[Dict[str, Any]] = []

        for cam_id in config.CAMERA_VIDEOS.keys():
            if cam_id in target_cameras:
                c_hits = self.qstore.search_similar(
                    query_vector=query_vector,
                    limit=50,
                    camera_id=cam_id
                )
                hits_by_camera[cam_id] = c_hits
                all_hits.extend(c_hits)
            else:
                hits_by_camera[cam_id] = []

        if not all_hits:
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
                "has_query_face": query_face_emb is not None,
                "message": "No candidates found in Qdrant database."
            }

        # 3. Two-Stage Candidate Re-Ranking & Independent Per-Camera Gating
        per_camera_best: Dict[str, Dict[str, Any]] = {}
        confirmed_camera_ids: List[str] = []
        all_evaluated_candidates: List[Dict[str, Any]] = []

        for cam_id in config.CAMERA_VIDEOS.keys():
            cam_hits = hits_by_camera.get(cam_id, [])
            if not cam_hits:
                continue

            # Evaluate top candidates in this camera with Stage 2 Face Verification
            evaluated_cam_candidates: List[Dict[str, Any]] = []
            best_bio_candidate: Optional[Dict[str, Any]] = None
            highest_face_sim = -1.0

            for hit_idx, hit in enumerate(cam_hits[:25]):
                fmt = self._format_hit(hit)
                face_eval = verify_crop_face(fmt.get("representative_crop_path"))
                fmt.update(face_eval)
                evaluated_cam_candidates.append(fmt)
                all_evaluated_candidates.append(fmt)

                # Check for confirmed biometric face match
                f_sim = face_eval.get("face_similarity")
                if face_eval.get("face_verdict") == "CONFIRMED_MATCH" and f_sim is not None:
                    if f_sim > highest_face_sim:
                        highest_face_sim = f_sim
                        best_bio_candidate = fmt

            # Calculate runner-up margin for top appearance candidates
            top_cam_hit = cam_hits[0]
            top_cam_score = float(top_cam_hit["score"])
            second_cam_score = float(cam_hits[1]["score"]) if len(cam_hits) > 1 else 0.0
            cam_margin = top_cam_score - second_cam_score

            # Check time co-occurrence between top 2 appearance candidates
            is_cooccurring = False
            competitor_track = None
            if len(cam_hits) > 1:
                p1 = top_cam_hit["payload"]
                p2 = cam_hits[1]["payload"]
                competitor_track = p2.get("track_id")
                start1, end1 = p1.get("start_time", 0.0), p1.get("end_time", 0.0)
                start2, end2 = p2.get("start_time", 0.0), p2.get("end_time", 0.0)
                is_cooccurring = max(start1, start2) <= min(end1, end2)

            # ── Selection Hierarchy for this Camera ──
            if best_bio_candidate is not None:
                # Priority 1: Biometrically Confirmed True Match!
                # Promoted over any bystanders who had similar clothes but no face visible
                best = best_bio_candidate
                best["status"] = "CONFIRMED_MATCH"
                best["is_confirmed"] = True
                best["status_label"] = f"Biometrically Confirmed Match (Face: {best['face_similarity']:.2f})"
                best["margin"] = round(cam_margin, 4)
                best["is_cooccurring"] = is_cooccurring
                best["competitor_track_id"] = competitor_track
                confirmed_camera_ids.append(cam_id)
            else:
                # Priority 2: Filter out bystanders vetoed by face mismatch
                valid_candidates = [
                    c for c in evaluated_cam_candidates
                    if c.get("face_verdict") != "VETOED_MISMATCH"
                ]

                if not valid_candidates:
                    # All candidates were vetoed bystanders
                    best = evaluated_cam_candidates[0]
                    best["status"] = "NOT_DETECTED"
                    best["is_confirmed"] = False
                    best["status_label"] = "Rejected (Face Mismatch)"
                    best["rejection_reason"] = "Bystanders wearing similar clothing vetoed by facial biometrics."
                else:
                    best = valid_candidates[0]
                    # Compute true runner-up margin among valid (non-vetoed) candidates
                    if len(valid_candidates) > 1:
                        cam_margin = float(valid_candidates[0]["score"]) - float(valid_candidates[1]["score"])
                        competitor_track = valid_candidates[1].get("track_id")
                    else:
                        cam_margin = float(valid_candidates[0]["score"])
                        competitor_track = None

                    best["margin"] = round(cam_margin, 4)
                    best["is_cooccurring"] = is_cooccurring
                    best["competitor_track_id"] = competitor_track

                    # Tier 1 Hardening: Strictly enforce ambiguity margin across all candidates.
                    # High similarity alone does not bypass margin checks (different people can score >= 0.85 in similar clothing).
                    effective_margin = max(ambiguity_margin, 0.04)

                    if best["score"] >= score_threshold:
                        if cam_margin >= effective_margin:
                            best["status"] = "CONFIRMED_MATCH"
                            best["is_confirmed"] = True
                            if best["score"] >= 0.85:
                                best["status_label"] = "Appearance Confirmed Match (High Confidence)"
                            else:
                                best["status_label"] = "Appearance Confirmed Match"
                            confirmed_camera_ids.append(cam_id)
                        else:
                            best["status"] = "AMBIGUOUS"
                            best["is_confirmed"] = False
                            best["status_label"] = f"Ambiguous Match (Margin {cam_margin:.3f} < {effective_margin:.2f})"
                    else:
                        best["status"] = "NOT_DETECTED"
                        best["is_confirmed"] = False
                        best["status_label"] = "Not Detected (Below Threshold)"

            per_camera_best[cam_id] = best

        # 4. Global top candidates and top-1 decision
        # Prioritize biometrically confirmed candidates at the top of the global results
        bio_matches = [c for c in all_evaluated_candidates if c.get("face_verdict") == "CONFIRMED_MATCH"]
        bio_matches.sort(key=lambda c: c.get("face_similarity", 0.0), reverse=True)

        appearance_matches = [c for c in all_evaluated_candidates if c.get("face_verdict") != "CONFIRMED_MATCH"]
        appearance_matches.sort(key=lambda c: c["score"], reverse=True)

        # Global top list: best biometric matches first, then top appearance hits
        global_candidates = bio_matches + appearance_matches

        # Top-1 candidate determination
        if bio_matches:
            top1 = bio_matches[0]
            top1_score = top1["score"]
            top2_score = bio_matches[1]["score"] if len(bio_matches) > 1 else (appearance_matches[0]["score"] if appearance_matches else 0.0)
            margin = round(abs(top1_score - top2_score), 4)

            is_reg, id_label = self.registry.is_track_registered(top1["track_id"])
            identity_display = id_label if is_reg else "Unregistered Person"

            decision = "CONFIRMED_MATCH"
            msg = (
                f"Biometrically confirmed match with track {top1['track_id']} "
                f"({identity_display}) at face similarity {top1['face_similarity']:.3f} (body score: {top1_score:.3f})."
            )
        else:
            top1 = global_candidates[0]
            top1_score = top1["score"]
            top2_score = global_candidates[1]["score"] if len(global_candidates) > 1 else 0.0
            margin = round(top1_score - top2_score, 4)

            is_reg, id_label = self.registry.is_track_registered(top1["track_id"])
            identity_display = id_label if is_reg else "Unregistered Person"

            effective_global_margin = max(ambiguity_margin, 0.04)
            if top1_score >= score_threshold:
                if margin >= effective_global_margin:
                    decision = "CONFIRMED_MATCH"
                    msg = (
                        f"Confident match with track {top1['track_id']} "
                        f"({identity_display}) at similarity {top1_score:.3f} (margin: {margin:.3f})."
                    )
                else:
                    decision = "AMBIGUOUS"
                    identity_display = "Unknown (Ambiguous Candidate)"
                    msg = (
                        f"Match candidate {top1['track_id']} exceeded threshold "
                        f"({top1_score:.3f}), but top-2 margin ({margin:.3f}) is below "
                        f"ambiguity threshold ({effective_global_margin:.2f})."
                    )
            else:
                decision = "UNKNOWN"
                identity_display = "Unknown"
                msg = (
                    f"Similarity score ({top1_score:.3f}) is below confidence threshold "
                    f"({score_threshold:.2f}). Reported as Unknown."
                )

        # Deduplicate candidates list
        seen_track_ids = set()
        dedup_candidates = []
        for c in global_candidates:
            if c["track_id"] not in seen_track_ids:
                seen_track_ids.add(c["track_id"])
                dedup_candidates.append(c)

        return {
            "decision": decision,
            "matched_track_id": top1["track_id"] if decision == "CONFIRMED_MATCH" else None,
            "camera_id": top1["camera_id"] if decision == "CONFIRMED_MATCH" else None,
            "identity_label": identity_display,
            "match_score": round(top1_score, 4),
            "margin": round(margin, 4),
            "top_candidate": dedup_candidates[0] if dedup_candidates else None,
            "candidates": dedup_candidates,
            "per_camera_best": per_camera_best,
            "confirmed_camera_ids": confirmed_camera_ids,
            "has_query_face": query_face_emb is not None,
            "query_face_info": {
                "confidence": query_face_info["confidence"],
                "bbox": query_face_info["bbox"]
            } if query_face_info else None,
            "message": msg
        }

