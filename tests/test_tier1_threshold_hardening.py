import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.matcher import ScreenshotMatcher
from src.identity_registry import IdentityRegistry

class MockQdrantStore:
    def __init__(self, hits_by_cam=None):
        self.hits_by_cam = hits_by_cam or {}

    def search_similar(self, query_vector, limit=50, camera_id=None):
        return self.hits_by_cam.get(camera_id, [])

class MockExtractor:
    def extract_features(self, crop):
        return np.ones((config.EMBEDDING_DIM,), dtype=np.float32) / np.sqrt(config.EMBEDDING_DIM)

class MockFaceModule:
    def detect_face(self, crop):
        return None
    def extract_embedding(self, crop, face_info=None):
        return None
    def verify_pair(self, query_feat, cand_feat):
        return {
            "verdict": "UNAVAILABLE",
            "similarity": None,
            "message": "Face unavailable"
        }

def run_tier1_hardening_tests():
    print("==================================================")
    print("RUNNING TIER 1 THRESHOLD & MARGIN HARDENING TESTS")
    print("==================================================")

    dummy_crop = np.zeros((128, 64, 3), dtype=np.uint8)

    # TEST CASE 1: Score >= 0.85 with narrow competitor margin (< 0.05)
    # In old code: this would auto-confirm because score >= 0.85!
    # In Tier 1 hardened code: it MUST be marked AMBIGUOUS because two people look identical!
    hits_narrow_high = {
        "Camera_1": [
            {
                "score": 0.880,
                "payload": {
                    "track_id": "Camera_1_101",
                    "camera_id": "Camera_1",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "representative_crop_path": "",
                    "source_video_path": ""
                }
            },
            {
                "score": 0.865, # margin = 0.015 (< 0.05)
                "payload": {
                    "track_id": "Camera_1_102",
                    "camera_id": "Camera_1",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "representative_crop_path": "",
                    "source_video_path": ""
                }
            }
        ]
    }

    matcher1 = ScreenshotMatcher(
        qstore=MockQdrantStore(hits_narrow_high),
        extractor=MockExtractor(),
        face_module=MockFaceModule()
    )

    res1 = matcher1.query_crop(dummy_crop, score_threshold=0.70, ambiguity_margin=0.05)
    print(f"Test 1 (Score=0.88, Margin=0.015): Decision={res1['decision']}, Per-Cam={res1['per_camera_best']['Camera_1']['status']}")
    assert res1["decision"] == "AMBIGUOUS", f"Expected AMBIGUOUS for narrow margin at 0.88 score, got {res1['decision']}"
    assert res1["per_camera_best"]["Camera_1"]["status"] == "AMBIGUOUS"
    print("[PASS] Test 1: 0.85 auto-confirm bypass successfully neutralized. Narrow margin correctly classified as AMBIGUOUS.")

    # TEST CASE 2: Non-co-occurring tracks across time
    # In old code: line 273 would drop margin to min(0.05, 0.01) = 0.01, falsely confirming margin=0.018!
    # In Tier 1 hardened code: full ambiguity margin is preserved.
    hits_non_cooccurring = {
        "Camera_1": [
            {
                "score": 0.760,
                "payload": {
                    "track_id": "Camera_1_201",
                    "camera_id": "Camera_1",
                    "start_time": 5.0,
                    "end_time": 15.0,
                    "duration": 10.0,
                    "representative_crop_path": "",
                    "source_video_path": ""
                }
            },
            {
                "score": 0.742, # margin = 0.018 (< 0.05)
                "payload": {
                    "track_id": "Camera_1_202",
                    "camera_id": "Camera_1",
                    "start_time": 45.0, # Completely different timestamp (not co-occurring)
                    "end_time": 55.0,
                    "duration": 10.0,
                    "representative_crop_path": "",
                    "source_video_path": ""
                }
            }
        ]
    }

    matcher2 = ScreenshotMatcher(
        qstore=MockQdrantStore(hits_non_cooccurring),
        extractor=MockExtractor(),
        face_module=MockFaceModule()
    )

    res2 = matcher2.query_crop(dummy_crop, score_threshold=0.70, ambiguity_margin=0.05)
    print(f"Test 2 (Non-cooccurring, Margin=0.018): Decision={res2['decision']}, Per-Cam={res2['per_camera_best']['Camera_1']['status']}")
    assert res2["decision"] == "AMBIGUOUS", f"Expected AMBIGUOUS for non-cooccurring narrow margin, got {res2['decision']}"
    assert res2["per_camera_best"]["Camera_1"]["status"] == "AMBIGUOUS"
    print("[PASS] Test 2: Ambiguity margin collapse bug (0.01) fixed. Non-cooccurring narrow margin correctly rejected.")

    # TEST CASE 3: Clear Winner with high score and healthy margin
    hits_clear_winner = {
        "Camera_1": [
            {
                "score": 0.820,
                "payload": {
                    "track_id": "Camera_1_301",
                    "camera_id": "Camera_1",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "representative_crop_path": "",
                    "source_video_path": ""
                }
            },
            {
                "score": 0.710, # margin = 0.110 (>= 0.05)
                "payload": {
                    "track_id": "Camera_1_302",
                    "camera_id": "Camera_1",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "representative_crop_path": "",
                    "source_video_path": ""
                }
            }
        ]
    }

    matcher3 = ScreenshotMatcher(
        qstore=MockQdrantStore(hits_clear_winner),
        extractor=MockExtractor(),
        face_module=MockFaceModule()
    )

    res3 = matcher3.query_crop(dummy_crop, score_threshold=0.70, ambiguity_margin=0.05)
    print(f"Test 3 (Clear Winner, Score=0.82, Margin=0.11): Decision={res3['decision']}, Per-Cam={res3['per_camera_best']['Camera_1']['status']}")
    assert res3["decision"] == "CONFIRMED_MATCH"
    assert res3["per_camera_best"]["Camera_1"]["status"] == "CONFIRMED_MATCH"
    print("[PASS] Test 3: Clear positive match correctly confirmed.")

    # TEST CASE 4: Low score below hardened threshold (0.68 < 0.70)
    # In old code: 0.68 >= 0.65 would be confirmed!
    # In Tier 1 hardened code: 0.68 < 0.70 is correctly marked UNKNOWN / NOT_DETECTED.
    hits_borderline = {
        "Camera_1": [
            {
                "score": 0.680, # Below 0.70
                "payload": {
                    "track_id": "Camera_1_401",
                    "camera_id": "Camera_1",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "representative_crop_path": "",
                    "source_video_path": ""
                }
            }
        ]
    }

    matcher4 = ScreenshotMatcher(
        qstore=MockQdrantStore(hits_borderline),
        extractor=MockExtractor(),
        face_module=MockFaceModule()
    )

    res4 = matcher4.query_crop(dummy_crop, score_threshold=config.DEFAULT_MATCH_THRESHOLD, ambiguity_margin=0.05)
    print(f"Test 4 (Borderline Score=0.68 < 0.70): Decision={res4['decision']}, Per-Cam={res4['per_camera_best']['Camera_1']['status']}")
    assert res4["decision"] == "UNKNOWN"
    assert res4["per_camera_best"]["Camera_1"]["status"] == "NOT_DETECTED"
    print("[PASS] Test 4: Borderline false-positive score (0.68) rejected by hardened threshold (0.70).")

    print("\n==================================================")
    print("ALL TIER 1 THRESHOLD & MARGIN HARDENING TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_tier1_hardening_tests()
