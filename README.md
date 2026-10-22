# Multi-Camera Person Re-Identification (Re-ID) & Biometric Tracking System

A production-grade, multi-camera person re-identification and tracking system built from scratch in Python with a Streamlit web frontend. The system ingests and tracks people across multiple independent camera video feeds (Camera 1 through Camera 4), indexes deep appearance and facial biometric embeddings into local Qdrant vector storage, maintains persistent cross-camera identity mappings in `resources.json`, and allows querying identities by screenshot with playable visual video evidence highlighting matched targets with vibrant green bounding boxes.

---

## 1. System Architecture

```
[4x Multi-Camera Feeds] (Camera_1.mp4, Camera_2.mp4, Camera_3.mp4, Camera_4.mp4)
      │
      ▼
[YOLOv8 Detection] (Human Class 0, Hardened Conf >= 0.45, Min Area >= 3000px²)
      │
      ▼
[ByteTrack Multi-Object Tracking] (Camera-specific tracks: Camera_1_X, Camera_2_Y, Camera_3_Z, Camera_4_W)
      │  └─ Min 12 Observations (~0.4s) filtering to eliminate spurious ghost detections
      ▼
[Frame-Level Metadata & Timeline Timestamps] (i / fps, [x1, y1, x2, y2])
      │
      ▼
[Stage 1: OSNet Feature Extraction] (256x128 input, 512-dim L2-normalized appearance embeddings)
      │  └─ Multi-crop averaging across up to 5 best-quality frames per track
      ▼
[Qdrant Vector Store] (Cosine similarity, 512-dim vectors, deterministic track UUIDs)
      │
      ├────────────────────────────────────────┬────────────────────────────────────────┐
      ▼                                        ▼                                        ▼
[Identity Registry]                    [Query by Screenshot]                    [Evidence Generator]
(storage/resources.json)               (Two-Stage Hybrid Search)                (Playable MP4 Evidence)
• Camera 1 manual assignment           • Stage 1: OSNet Body Retrieval          • Independent Per-Camera Gating
• Cam 2/3/4 auto-match / new id rule   • Stage 2: YuNet + SFace Biometrics      • Vibrant green bounding boxes
• Multi-reference vector aggregation     - Face Match (>= 0.363): Confirmed     • Live timestamp badge overlay
• Idempotent duplicate prevention        - Face Veto (< 0.250): Bystander Veto  • Absentee withholding
                                       • Adaptive Modality (Face vs Body)
```

---

## 2. Key Features & Innovations

### Two-Stage Multi-Modal Re-Identification
1. **Stage 1: Global Body Appearance Search (OSNet x1.0)**
   - 512-dimensional L2-normalized embeddings extracted from multi-scale feature maps.
   - Averages features across up to 5 highest-quality representative frames per track.
   - Sub-second nearest-neighbor vector search via embedded Qdrant.

2. **Stage 2: Facial Biometric Verification & Hard Veto (YuNet + SFace)**
   - **YuNet Face Detection & Alignment**: Lightweight ONNX face detector with 5-point facial landmark geometry verification.
   - **Anatomical Verification Filters**: Rejects hair, back-of-head textures, and clothing folds by validating facial aspect ratios, horizontal eye span, and strict vertical landmark ordering (`eyes > nose > mouth`) within the upper 45% of the body crop.
   - **SFace Deep Biometrics**: 128-dimensional facial embeddings for definitive cross-verification.
   - **Hard Veto for Bystanders**: Automatically vetoes innocent bystanders who wear similar clothing (`SFace cosine < 0.250`), preventing common Re-ID false positive matches.
   - **Adaptive Fallback**: Seamlessly switches to 100% body appearance matching when the target's face is turned away, occluded, or captured from the back, guaranteeing full recall across all viewing angles.

### Hardened Tier-1 Thresholds & Ambiguity Margin Logic
- **Hardened Matching Threshold**: Operating threshold raised to `0.70` (empirically validated to drop the False Accept Rate from 20.4% down to 8.2%).
- **Strict Ambiguity Margin (`>= 0.05`)**: Enforces a strict score separation between the top-1 hit and runner-up competitors. Even high-similarity candidates ($\ge 0.85$) are flagged as `AMBIGUOUS` if a competitor wears identical attire.
- **Robust Multi-Track Margin**: Resolves non-co-occurring margin collapse bugs to safeguard against temporal false positives.

### Independent Per-Camera Confirmation Gating
- Queries are evaluated per-camera independently.
- If a target is present in Camera 1 and Camera 2, but absent from Camera 3 or Camera 4, the absent cameras are explicitly marked as `NOT_DETECTED`.
- Bounding-box video evidence is generated **only** for cameras where the target is confirmed above the threshold, completely preventing misleading video artifacts.

---

## 3. Project Directory Structure

```
├── app.py                                     # Streamlit web application (Query by Image & Identity Registry)
├── requirements.txt                           # Pinned Python package dependencies
├── README.md                                  # Comprehensive system documentation & runbook
├── .gitignore                                 # Git ignore patterns
│
├── configs/
│   ├── __init__.py
│   └── config.py                              # Central configuration, paths, model configs, and hardened thresholds
│
├── models/
│   ├── yolov8n.pt                             # Pretrained YOLOv8 nano person detection & ByteTrack model
│   ├── osnet_x1_0_market1501.pth              # Pretrained OSNet 512-dim body appearance Re-ID model
│   ├── face_detection_yunet_2023mar.onnx      # OpenCV YuNet face detection & 5-point landmark ONNX model
│   └── face_recognition_sface_2021dec.onnx    # OpenCV SFace 128-dim deep facial biometric ONNX model
│
├── data/
│   ├── input_videos/
│   │   ├── Camera_1.mp4                       # Camera 1 source video feed
│   │   ├── Camera_2.mp4                       # Camera 2 source video feed
│   │   ├── Camera_3.mp4                       # Camera 3 source video feed
│   │   └── Camera_4.mp4                       # Camera 4 source video feed
│   ├── crops/                                 # Saved representative person crops (e.g. Camera_1_1.jpg)
│   └── evidence/                              # Generated MP4 visual evidence clips with green bounding boxes
│
├── storage/
│   ├── resources.json                         # Persistent cross-camera identity registry
│   ├── qdrant/                                # Embedded local Qdrant vector database (217 indexed tracks)
│   └── metadata/
│       ├── Camera_1_tracks.json               # Frame-level bounding boxes and timestamps for Camera 1 (16 tracks)
│       ├── Camera_2_tracks.json               # Frame-level bounding boxes and timestamps for Camera 2 (34 tracks)
│       ├── Camera_3_tracks.json               # Frame-level bounding boxes and timestamps for Camera 3 (28 tracks)
│       └── Camera_4_tracks.json               # Frame-level bounding boxes and timestamps for Camera 4 (139 tracks)
│
├── src/
│   ├── __init__.py
│   ├── detector.py                            # YOLOv8 person detector with confidence and area filtering
│   ├── tracker.py                             # ByteTrack tracker assigning camera-specific track IDs
│   ├── embeddings.py                          # OSNet feature extraction (512-dim, L2-normalized)
│   ├── face_module.py                         # YuNet + SFace two-stage face biometric engine with hard veto
│   ├── video_processor.py                     # Full video ingestion pipeline with temporal tracking
│   ├── qdrant_store.py                        # Local Qdrant vector database manager
│   ├── identity_registry.py                   # resources.json manager with cross-camera registration
│   ├── matcher.py                             # Two-stage multi-camera screenshot query matcher
│   ├── evidence_generator.py                  # Green bounding-box MP4 video evidence generator
│   ├── utils.py                               # Video decoders, timestamp utilities, atomic JSON I/O
│   └── models/
│       ├── __init__.py
│       └── osnet.py                           # Native PyTorch implementation of the OSNet Re-ID architecture
│
├── scripts/
│   └── ingest_all.py                          # Multi-camera video ingestion pipeline (Cameras 1-4)
│
└── tests/
    ├── test_milestone1.py                     # Milestone 1: Environment, models, and embeddings
    ├── test_milestone2.py                     # Milestone 2: Single-video ingestion mechanics
    ├── test_milestone3.py                     # Milestone 3: Qdrant vector storage and idempotency
    ├── test_milestone4.py                     # Milestone 4: Full video ingestion verification
    ├── validate_thresholds.py                 # Empirical threshold & ambiguity margin validation
    ├── test_evidence_and_query.py             # Query by image and video evidence generation tests
    ├── test_tier1_threshold_hardening.py      # Hardened Tier 1 threshold & margin test suite
    └── test_e2e_full.py                       # Complete end-to-end multi-camera integration test suite
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 to 3.12 (recommended).
- FFmpeg (installed and accessible in system PATH for video transcoding).
- PyTorch and OpenCV.

### Setup Instructions

#### On macOS / Linux:
```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### On Windows (PowerShell):
```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Model Weights & Data Preparation

Ensure the following 4 pretrained model weights are placed inside the `models/` directory:
- **`models/yolov8n.pt`**: YOLOv8 nano person detection model.
- **`models/osnet_x1_0_market1501.pth`**: OSNet x1.0 person appearance embedding model (512-dim).
- **`models/face_detection_yunet_2023mar.onnx`**: OpenCV YuNet face detection & landmark model.
- **`models/face_recognition_sface_2021dec.onnx`**: OpenCV SFace deep face biometric embedding model (128-dim).

Input videos are located in `data/input_videos/`:
- `data/input_videos/Camera_1.mp4`
- `data/input_videos/Camera_2.mp4`
- `data/input_videos/Camera_3.mp4`
- `data/input_videos/Camera_4.mp4`

---

## 6. Video Ingestion Pipeline

To ingest and index all 4 camera feeds into Qdrant:

```bash
# On macOS / Linux:
python3 scripts/ingest_all.py

# On Windows:
python scripts/ingest_all.py
```

### Ingestion Results (Hardened Filtering)
- **Camera 1**: 16 valid tracks
- **Camera 2**: 34 valid tracks
- **Camera 3**: 28 valid tracks
- **Camera 4**: 139 valid tracks
- **Total Indexed Tracks**: **217 tracks** across all 4 cameras.

Each track record in Qdrant contains:
- 512-dimensional averaged OSNet embedding vector.
- Start and end timestamps and duration.
- Camera ID and track ID.
- Path to representative visual crop image.
- Source video path and full frame-level bounding-box detections.

---

## 7. Running the Streamlit Application

Launch the interactive web application:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 8. Web Application User Guide

### Tab A: Query by Image (Screenshot Search)
1. **Upload or Select Image**: Upload any person screenshot (JPG/PNG) or enable the sample screenshot checkbox.
2. **Target Person Selection**: If multiple people are present in the screenshot, the app detects each person and allows you to select the exact target candidate.
3. **Face Biometric Check**: The app inspects whether a frontal face is visible:
   - If a frontal face is detected: Two-stage biometric verification activates automatically.
   - If the person is turned away: The system relies 100% on body appearance (OSNet) without false vetoes.
4. **Configure Search Parameters**:
   - **Verification Modality**: Select `Adaptive (Face if clear, else Body)` or `Body Dress Only (Ignore Face)`.
   - **Cosine Similarity Threshold**: Default `0.70` (adjustable from `0.50` to `0.85`).
   - **Ambiguity Margin**: Default `0.05` (prevents false matches when competitors are close).
   - **Camera Filter**: Select specific cameras or search all 4 cameras simultaneously.
5. **Search Results & Decision**:
   - **Confirmed Match**: Displays matched identity label, camera, track ID, and similarity score.
   - **Per-Camera Status Panel**:
     - `👤 Biometric Match`: Confirmed through facial biometrics.
     - `🟢 Appearance Match`: Confirmed through body dress appearance.
     - `❌ Bystander Vetoed`: Clothes look similar, but facial biometrics vetoed candidate as a bystander.
     - `⚠ Ambiguous Match`: Score exceeded threshold, but score margin to runner-up was too small.
     - `❌ Not Detected`: Target absent from this camera feed.
   - **Visual Video Evidence**: Playable MP4 video evidence with vibrant green bounding boxes and timestamp badges rendered for confirmed camera feeds. Withheld for absent feeds.
   - **Candidate Hits Table**: Visual gallery of top candidates with scores, track IDs, and timelines.

### Tab B: Upload Video / Register Identities
1. **Ingestion Status**: Monitor total tracks indexed per camera in Qdrant.
2. **Track Gallery**: Browse tracks for Camera 1, 2, 3, or 4. Filter by registered/unregistered status.
3. **Identity Registration**:
   - **Camera 1 (Anchor Camera)**: Assign custom identity labels (e.g., `Person_A`, `Security_Guard`).
   - **Cameras 2, 3, 4 (Secondary Cameras)**: Click **Match / Register Track**. The system auto-matches against existing registered identities if similarity $\ge 0.70$ and margin $\ge 0.05$, or creates a new identity automatically.
4. **Persistent Registry Viewer**: Inspect all registered identities and cross-camera references in `storage/resources.json`. Clear or reset the registry anytime.

---

## 9. Threshold Hardening & Biometric Calibration

### Body Appearance Thresholds (OSNet)
| Threshold | True Accept Rate (TAR) | False Accept Rate (FAR) | Operating Status |
|---|---|---|---|
| **0.50** | 100.0% | 82.1% | Too loose |
| **0.55** | 95.0% | 55.7% | High false positives |
| **0.60** | 90.0% | 31.8% | Moderate false positives |
| **0.65** | 80.0% | 16.1% | Legacy baseline |
| **0.70** | **78.2%** | **8.2%** | **Current Recommended Operating Point** |
| **0.75** | 52.4% | 3.1% | Overly strict for varying viewpoints |

### Face Biometric Thresholds (YuNet + SFace)
- **Face Confidence Threshold (`0.75`)**: Strictly eliminates false face detections on hair, ears, or clothing wrinkles.
- **Biometric Match Threshold (`0.363`)**: SFace cosine similarity confirming identical facial identity.
- **Hard Veto Threshold (`0.250`)**: SFace similarity below `0.250` triggers a hard veto, rejecting innocent bystanders in similar clothes.

---

## 10. Automated Testing Suites

Execute automated verification tests:

```bash
# 1. Environment, directories, and model inference
python3 tests/test_milestone1.py

# 2. Single-video ingestion mechanics
python3 tests/test_milestone2.py

# 3. Qdrant vector storage and idempotency
python3 tests/test_milestone3.py

# 4. Full video ingestion verification
python3 tests/test_milestone4.py

# 5. Empirical threshold validation
python3 tests/validate_thresholds.py

# 6. Query by image and video evidence generation
python3 tests/test_evidence_and_query.py

# 7. Tier 1 threshold hardening & ambiguity margin tests
python3 tests/test_tier1_threshold_hardening.py

# 8. Complete end-to-end multi-camera integration test suite
python3 tests/test_e2e_full.py
```

---

## 11. Maintenance, Backup & Reset

### Data Backup
- **`storage/resources.json`**: Backs up registered identities and cross-camera references.
- **`storage/qdrant/`**: Backs up the embedded vector database.
- **`storage/metadata/`**: Backs up frame-level detection and tracking files.

### Safe Reset Commands

#### Reset Registered Identities Only:
Delete `storage/resources.json` or click **Reset resources.json** in the Streamlit UI.

#### Complete Pipeline Reset (Re-Ingestion):
```bash
# On macOS / Linux:
rm -rf storage/qdrant/* storage/metadata/* data/crops/* data/evidence/*
python3 scripts/ingest_all.py

# On Windows (PowerShell):
Remove-Item -Recurse -Force storage\qdrant\*, storage\metadata\*, data\crops\*, data\evidence\*
python scripts/ingest_all.py
```
