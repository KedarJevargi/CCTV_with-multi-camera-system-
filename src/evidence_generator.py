import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

import configs.config as config
from src.utils import load_json, format_timestamp, draw_bounding_box

def _get_ffmpeg_bin() -> Optional[str]:
    """Resolves a working ffmpeg binary using imageio_ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")

class EvidenceGenerator:
    """
    Generates annotated video clips and visual evidence with vibrant green bounding boxes
    around the matched person throughout their observed appearances.
    """
    def __init__(
        self,
        metadata_dir: Union[str, Path] = config.METADATA_DIR,
        evidence_dir: Union[str, Path] = config.EVIDENCE_DIR
    ):
        self.metadata_dir = Path(metadata_dir)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._cached_meta: Dict[str, Dict[str, Any]] = {}

    def _ensure_faststart_h264(self, video_path: Path) -> bool:
        """Ensures an MP4 video is encoded with H.264 (yuv420p) and +faststart for browser playback."""
        ffmpeg_bin = _get_ffmpeg_bin()
        if not ffmpeg_bin or not video_path.exists() or video_path.stat().st_size == 0:
            return False
        
        # Check if moov atom is in the first 8192 bytes and contains avc1 (H.264) codec
        try:
            with open(video_path, "rb") as f:
                head = f.read(8192)
                if b"moov" in head and b"avc1" in head:
                    return True
        except Exception:
            pass

        tmp_out = video_path.parent / f"{video_path.stem}_transcoded.mp4"
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(tmp_out)
        ]
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
                tmp_out.replace(video_path)
                return True
        except Exception:
            pass
        return False

    def _get_camera_metadata(self, camera_id: str) -> Dict[str, Any]:
        """
        Loads and caches metadata for a camera.
        """
        if camera_id not in self._cached_meta:
            meta_path = self.metadata_dir / f"{camera_id}_tracks.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file missing for {camera_id}: {meta_path}")
            self._cached_meta[camera_id] = load_json(meta_path)
        return self._cached_meta[camera_id]

    def generate_annotated_evidence(
        self,
        track_id: str,
        display_label: Optional[str] = None,
        max_frames_to_render: int = 150,
        box_color: Tuple[int, int, int] = (0, 255, 0),  # Bright Green (BGR)
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieves stored frame-level detections for the track, opens the source video,
        and renders green bounding boxes on every appearance frame.
        Outputs:
        - A playable MP4 evidence video in data/evidence/
        - Keyframe snapshots showing the person at early, middle, and late stages.
        """
        # Dynamically resolve camera_id from the track_id prefix (supports N cameras)
        camera_id = None
        for cam in config.CAMERA_VIDEOS.keys():
            if track_id.startswith(f"{cam}_"):
                camera_id = cam
                break
        if camera_id is None:
            raise ValueError(f"Cannot determine camera for track '{track_id}'. Known cameras: {list(config.CAMERA_VIDEOS.keys())}")
        meta = self._get_camera_metadata(camera_id)

        if track_id not in meta:
            raise KeyError(f"Track '{track_id}' not found in {camera_id} metadata.")

        track_info = meta[track_id]
        video_path = track_info["source_video_path"]
        observations = track_info["frame_observations"]

        if not observations:
            raise ValueError(f"No frame observations recorded for track '{track_id}'.")

        # Output paths
        output_filename = f"{track_id}_evidence.mp4"
        output_path = self.evidence_dir / output_filename
        raw_output_path = self.evidence_dir / f"{track_id}_raw.mp4"

        obs_map = {obs["frame_idx"]: obs for obs in observations}
        frame_indices = sorted(obs_map.keys())

        # If already rendered and not forced, ensure faststart H.264 and return immediately
        if not force_regenerate and output_path.exists() and output_path.stat().st_size > 0:
            self._ensure_faststart_h264(output_path)
            existing_keyframes = []
            for kf_file in sorted(self.evidence_dir.glob(f"{track_id}_kf_*.jpg")):
                stem = kf_file.stem
                try:
                    f_idx = int(stem.split("_kf_")[-1])
                    ts = obs_map.get(f_idx, {}).get("timestamp", 0.0)
                except Exception:
                    f_idx = 0
                    ts = 0.0
                existing_keyframes.append({
                    "frame_idx": f_idx,
                    "timestamp": ts,
                    "image_path": str(kf_file).replace("\\", "/")
                })
            if len(existing_keyframes) >= 2:
                return {
                    "track_id": track_id,
                    "camera_id": camera_id,
                    "display_label": display_label or track_id,
                    "start_time": track_info["start_time"],
                    "end_time": track_info["end_time"],
                    "duration": track_info["duration"],
                    "total_frames_observed": track_info["total_frames_observed"],
                    "evidence_video_path": str(output_path).replace("\\", "/"),
                    "keyframes": existing_keyframes,
                    "source_video_path": video_path
                }

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Source video could not be opened: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try avc1 (native H.264) first, fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))

        # Subsample frames if track is extremely long (to prevent oversized video files)
        stride = 1
        if len(frame_indices) > max_frames_to_render:
            stride = max(1, len(frame_indices) // max_frames_to_render)
            selected_indices = frame_indices[::stride]
        else:
            selected_indices = frame_indices

        annotated_keyframes: List[Dict[str, Any]] = []
        keyframe_step = max(1, len(selected_indices) // 4)

        try:
            # Fast sequential decoding: seek once to first frame, then cap.grab() forward
            cap.set(cv2.CAP_PROP_POS_FRAMES, selected_indices[0])
            current_fidx = selected_indices[0]

            for idx_count, target_fidx in enumerate(selected_indices):
                while current_fidx < target_fidx:
                    cap.grab()
                    current_fidx += 1
                ret, frame = cap.read()
                current_fidx += 1
                if not ret:
                    continue

                obs = obs_map[target_fidx]
                bbox = obs["bbox"]
                timestamp = obs["timestamp"]
                conf = obs["confidence"]

                # Badge label on green bounding box
                badge_text = f"{display_label or track_id} | {format_timestamp(timestamp)} | {conf:.2f}"
                annotated = draw_bounding_box(
                    image=frame,
                    bbox=bbox,
                    label=badge_text,
                    color=box_color,
                    thickness=3
                )

                # Write to output video
                out.write(annotated)

                # Save 3-4 keyframe snapshots for quick UI image display
                if idx_count % keyframe_step == 0 and len(annotated_keyframes) < 4:
                    keyframe_name = f"{track_id}_kf_{target_fidx}.jpg"
                    keyframe_path = self.evidence_dir / keyframe_name
                    cv2.imwrite(str(keyframe_path), annotated)
                    annotated_keyframes.append({
                        "frame_idx": target_fidx,
                        "timestamp": timestamp,
                        "image_path": str(keyframe_path).replace("\\", "/")
                    })

        finally:
            cap.release()
            out.release()

        # Transcode with imageio_ffmpeg for 100% browser compatibility (H.264 + faststart)
        ffmpeg_bin = _get_ffmpeg_bin()
        if ffmpeg_bin:
            ffmpeg_cmd = [
                ffmpeg_bin, "-y",
                "-i", str(raw_output_path),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path)
            ]
            try:
                p = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if p.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    if raw_output_path.exists():
                        raw_output_path.unlink()
                else:
                    if raw_output_path.exists():
                        raw_output_path.replace(output_path)
            except Exception:
                if raw_output_path.exists():
                    raw_output_path.replace(output_path)
        else:
            if raw_output_path.exists():
                raw_output_path.replace(output_path)

        return {
            "track_id": track_id,
            "camera_id": camera_id,
            "display_label": display_label or track_id,
            "start_time": track_info["start_time"],
            "end_time": track_info["end_time"],
            "duration": track_info["duration"],
            "total_frames_observed": track_info["total_frames_observed"],
            "evidence_video_path": str(output_path).replace("\\", "/"),
            "keyframes": annotated_keyframes,
            "source_video_path": video_path
        }
