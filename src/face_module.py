from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import cv2
import numpy as np

import configs.config as config

class FaceBiometricModule:
    """
    Two-Stage Biometric Verification Engine:
    - Stage 2A: Ultra-fast lightweight face detection and 5-point landmark alignment via OpenCV YuNet (ONNX).
    - Stage 2B: 128-dimensional Deep Biometric embedding extraction and cosine verification via SFace (ONNX).
    """
    def __init__(
        self,
        yunet_path: Union[str, Path] = config.YUNET_MODEL_PATH,
        sface_path: Union[str, Path] = config.SFACE_MODEL_PATH,
        conf_threshold: float = config.FACE_CONF_THRESHOLD,
        match_threshold: float = config.FACE_MATCH_THRESHOLD,
        mismatch_threshold: float = config.FACE_MISMATCH_THRESHOLD
    ):
        self.yunet_path = Path(yunet_path)
        self.sface_path = Path(sface_path)
        self.conf_threshold = conf_threshold
        self.match_threshold = match_threshold
        self.mismatch_threshold = mismatch_threshold

        if not self.yunet_path.exists():
            raise FileNotFoundError(f"YuNet model missing at {self.yunet_path}")
        if not self.sface_path.exists():
            raise FileNotFoundError(f"SFace model missing at {self.sface_path}")

        # Initialize YuNet detector with a default input size (updated dynamically on inference)
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(self.sface_path),
            config=""
        )

    def detect_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects the most prominent human face in a person crop or image.
        Returns a dict containing face bounding box, 5-point landmarks, confidence, and raw face array.
        Returns None if no face exceeds confidence threshold.
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        if w < 10 or h < 10:
            return None

        # Dynamically set input dimensions for YuNet
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None

        # Sort by confidence descending
        faces = sorted(faces, key=lambda f: float(f[-1]), reverse=True)
        best_face = faces[0]
        conf = float(best_face[-1])

        if conf < self.conf_threshold:
            return None

        bbox = [float(v) for v in best_face[:4]] # [x, y, w, h]
        landmarks = [float(v) for v in best_face[4:14]] # [rx, ry, lx, ly, nx, ny, rmx, rmy, lmx, lmy]

        fx, fy, fw, fh = bbox
        # 1. Face location check: Human face must be located in the upper 45% of the person crop
        if fy > 0.45 * h:
            return None

        # 2. Minimum face dimension check
        if fw < 12 or fh < 12:
            return None

        # 3. Face Aspect Ratio: Frontal/semi-frontal face aspect ratio is between 0.5 and 1.5
        if not (0.5 <= (fw / fh) <= 1.5):
            return None

        # 4. 5-point Landmark Biometric Geometry Checks
        # YuNet format: [rx, ry, lx, ly, nx, ny, rmx, rmy, lmx, lmy]
        rx, ry = landmarks[0], landmarks[1]
        lx, ly = landmarks[2], landmarks[3]
        nx, ny = landmarks[4], landmarks[5]
        rmx, rmy = landmarks[6], landmarks[7]
        lmx, lmy = landmarks[8], landmarks[9]

        eye_dx = abs(lx - rx)
        eye_dy = abs(ly - ry)
        eye_dist = (eye_dx**2 + eye_dy**2)**0.5

        if eye_dist < 6.0:
            return None

        # Reject vertically oriented eye landmarks (e.g. neck folds, clothing collars, or hair parting)
        if eye_dx > 0 and (eye_dy / eye_dx) > 0.85:
            return None

        # 5. Natural Vertical Ordering: Eyes must be above Nose, and Nose must be above Mouth
        eye_y_avg = (ry + ly) / 2.0
        mouth_y_avg = (rmy + lmy) / 2.0

        if (mouth_y_avg - eye_y_avg) < 6.0:
            # Eyes and mouth collapsed onto same horizontal level (texture artifact)
            return None

        if not (eye_y_avg < ny < mouth_y_avg):
            # Nose is not vertically between eyes and mouth (scrambled landmarks on hair/back of head)
            return None

        # 6. Mouth horizontal span check
        mouth_dx = abs(lmx - rmx)
        if mouth_dx < 4.0:
            return None

        # Extract aligned face crop thumbnail for visualization
        try:
            aligned_thumb = self.recognizer.alignCrop(image, best_face)
        except Exception:
            aligned_thumb = None

        return {
            "bbox": bbox,
            "landmarks": landmarks,
            "confidence": round(conf, 4),
            "raw_face": best_face,
            "aligned_thumbnail": aligned_thumb
        }

    def extract_embedding(
        self,
        image: np.ndarray,
        face_info: Optional[Dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """
        Aligns face and extracts 128-dimensional L2-normalized biometric embedding.
        If face_info is not provided, runs detect_face first.
        Returns float32 numpy array of shape (128,) or None if face not found.
        """
        if face_info is None:
            face_info = self.detect_face(image)

        if face_info is None or "raw_face" not in face_info:
            return None

        try:
            aligned = self.recognizer.alignCrop(image, face_info["raw_face"])
            feat = self.recognizer.feature(aligned) # shape (1, 128)
            feat = feat.flatten().astype(np.float32)
            # Ensure L2 normalization
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            return feat
        except Exception:
            return None

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Computes cosine similarity between two 128-dimensional biometric embeddings.
        Values range from -1.0 to 1.0 (typical match >= 0.363).
        """
        if feat1 is None or feat2 is None:
            return 0.0

        f1 = np.asarray(feat1, dtype=np.float32).reshape(1, -1)
        f2 = np.asarray(feat2, dtype=np.float32).reshape(1, -1)

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        f1_norm = f1 / norm1
        f2_norm = f2 / norm2
        sim = float(np.dot(f1_norm, f2_norm.T)[0, 0])
        return round(sim, 4)

    def verify_pair(
        self,
        query_feat: Optional[np.ndarray],
        cand_feat: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluates biometric compatibility between query and candidate face embeddings.
        Implements Hard Veto rule:
        - If similarity < mismatch_threshold (0.25): Hard Veto (MISMATCH)
        - If similarity >= match_threshold (0.363): Confirmed (MATCH)
        - If in between: Ambiguous face
        - If either face is missing: UNAVAILABLE
        """
        if query_feat is None or cand_feat is None:
            return {
                "verdict": "UNAVAILABLE",
                "similarity": None,
                "is_match": None,
                "is_mismatch": False,
                "message": "Face biometric unavailable (face turned away or occluded)."
            }

        sim = self.compute_similarity(query_feat, cand_feat)

        if sim < self.mismatch_threshold:
            return {
                "verdict": "VETOED_MISMATCH",
                "similarity": sim,
                "is_match": False,
                "is_mismatch": True,
                "message": f"Biometric face mismatch (similarity {sim:.3f} < {self.mismatch_threshold:.2f}). Candidate is an innocent bystander in similar clothes."
            }
        elif sim >= self.match_threshold:
            return {
                "verdict": "CONFIRMED_MATCH",
                "similarity": sim,
                "is_match": True,
                "is_mismatch": False,
                "message": f"Biometric face match confirmed (similarity {sim:.3f} >= {self.match_threshold:.2f})."
            }
        else:
            return {
                "verdict": "AMBIGUOUS_FACE",
                "similarity": sim,
                "is_match": False,
                "is_mismatch": False,
                "message": f"Face similarity {sim:.3f} is in the borderline zone [{self.mismatch_threshold:.2f}, {self.match_threshold:.2f}]."
            }
