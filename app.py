import os
import sys
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import configs.config as config
from src.embeddings import OSNetFeatureExtractor
from src.detector import PersonDetector
from src.qdrant_store import QdrantStore
from src.identity_registry import IdentityRegistry
from src.matcher import ScreenshotMatcher
from src.evidence_generator import EvidenceGenerator
from src.utils import load_json, format_timestamp

# Page Configuration
st.set_page_config(
    page_title="Multi-Camera Person Re-Identification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .badge-confirmed {
        background-color: #2e7d32;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-ambiguous {
        background-color: #ed6c02;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-unknown {
        background-color: #d32f2f;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Cached Resource Singletons to avoid reloading weights on every interaction
@st.cache_resource
def get_feature_extractor():
    return OSNetFeatureExtractor()

@st.cache_resource
def get_person_detector():
    return PersonDetector()

@st.cache_resource
def get_qdrant_store():
    return QdrantStore()

@st.cache_resource
def get_identity_registry():
    return IdentityRegistry()

@st.cache_resource
def get_evidence_generator():
    return EvidenceGenerator()

@st.cache_resource
def get_screenshot_matcher():
    extractor = get_feature_extractor()
    detector = get_person_detector()
    qstore = get_qdrant_store()
    registry = get_identity_registry()
    return ScreenshotMatcher(
        qstore=qstore,
        extractor=extractor,
        detector=detector,
        registry=registry
    )

# Sidebar Navigation
st.sidebar.title("Navigation")
workflow_mode = st.sidebar.radio(
    "Select Workflow:",
    ["Query by Image", "Upload Video / Register Identities"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
qstore = get_qdrant_store()
registry = get_identity_registry()
registry.load()

try:
    total_tracks_db = qstore.count_tracks()
    st.sidebar.success(f"**Qdrant Tracks**: {total_tracks_db}")
    cam_counts = []
    for cam_id in config.CAMERA_VIDEOS.keys():
        n = qstore.count_tracks(cam_id)
        cam_counts.append(f"{cam_id}: {n}")
    st.sidebar.caption(" | ".join(cam_counts))
except Exception as e:
    st.sidebar.error("Qdrant store offline or locked")
    total_tracks_db = 0

registered_identities = len(registry.identities)
st.sidebar.info(f"**Registered Identities**: {registered_identities}")

st.sidebar.markdown("---")
st.sidebar.caption("Multi-Camera Person Re-ID Demo • Powered by OSNet & YOLO")


# ==============================================================================
# OPTION A: QUERY BY IMAGE
# ==============================================================================
if workflow_mode == "Query by Image":
    st.markdown('<div class="main-header">🎯 Query by Image (Screenshot Search)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a screenshot from any camera to retrieve candidate tracks, match identities, and view video evidence with green bounding boxes.</div>', unsafe_allow_html=True)

    if total_tracks_db == 0:
        st.warning("⚠️ Qdrant database is currently empty. Please switch to 'Upload Video / Register Identities' to verify ingestion first.")

    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader("Upload Person Screenshot (JPG / PNG):", type=["jpg", "jpeg", "png"])
        use_sample = st.checkbox("Or use sample screenshot from video data", value=False)
        sample_path = config.DATA_DIR / "test_screenshot.jpg"

    selected_image = None
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        selected_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif use_sample and sample_path.exists():
        selected_image = cv2.imread(str(sample_path))

    if selected_image is not None:
        matcher = get_screenshot_matcher()
        h, w = selected_image.shape[:2]

        with col_preview:
            st.image(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB), caption=f"Uploaded Image ({w}x{h})", use_container_width=True)

        st.markdown("---")
        st.subheader("Step 1: Person Candidate Selection")

        # Detect candidate persons if full frame
        crops_data = matcher.process_screenshot_crops(selected_image)

        if len(crops_data) > 1:
            st.info(f"Detected **{len(crops_data)}** person candidates in the screenshot. Select the person you wish to identify:")
            crop_cols = st.columns(min(len(crops_data), 5))
            crop_options = {}
            for idx, c in enumerate(crops_data):
                with crop_cols[idx % 5]:
                    rgb_crop = cv2.cvtColor(c["crop_image"], cv2.COLOR_BGR2RGB)
                    st.image(rgb_crop, caption=f"Person #{c['crop_id']} (conf: {c['confidence']:.2f})", use_container_width=True)
                    crop_options[f"Person #{c['crop_id']}"] = c["crop_image"]

            chosen_crop_label = st.selectbox("Select Target Candidate:", list(crop_options.keys()))
            query_crop_image = crop_options[chosen_crop_label]
        else:
            query_crop_image = crops_data[0]["crop_image"]
            st.success("Target person crop prepared for appearance matching.")

        st.markdown("---")
        st.subheader("Step 2: Identification & Evidence")

        col_btn, col_thresh, col_filter = st.columns([1, 2, 2])
        with col_thresh:
            threshold = st.slider("Cosine Similarity Threshold:", min_value=0.50, max_value=0.85, value=config.DEFAULT_MATCH_THRESHOLD, step=0.01)
        with col_filter:
            all_cams = list(config.CAMERA_VIDEOS.keys())
            camera_filter = st.multiselect(
                "Filter cameras:",
                options=all_cams,
                default=all_cams,
                help="Restrict search to specific cameras. All selected by default."
            )
        with col_btn:
            run_query = st.button("🔍 Search Person in Qdrant", type="primary")

        num_cameras = len(config.CAMERA_VIDEOS)
        if "query_result" not in st.session_state:
            st.session_state["query_result"] = None

        if run_query:
            with st.spinner("Extracting OSNet 512-dim embedding and querying Qdrant..."):
                query_result = matcher.query_crop(
                    crop_image=query_crop_image,
                    top_k=num_cameras * 4,
                    score_threshold=threshold,
                    ambiguity_margin=config.AMBIGUITY_THRESHOLD,
                    camera_filter=camera_filter if camera_filter else None
                )
                st.session_state["query_result"] = query_result
        else:
            query_result = st.session_state.get("query_result")

        if query_result:
            decision = query_result["decision"]
            match_score = query_result["match_score"]
            margin = query_result["margin"]
            top_cand = query_result["top_candidate"]
            per_camera_best = query_result.get("per_camera_best", {})
            active_cameras = [c for c in (camera_filter if camera_filter else config.CAMERA_VIDEOS.keys()) if c in per_camera_best]

            # Display Matching Decision Banner
            st.markdown("### Matching Decision")
            if decision == "CONFIRMED_MATCH":
                st.markdown(f'<div class="badge-confirmed">✔ CONFIRMED MATCH: {query_result["identity_label"]} (Top Score: {match_score:.3f}, Margin: {margin:.3f})</div>', unsafe_allow_html=True)
                st.success(query_result["message"])
            elif decision == "AMBIGUOUS":
                st.markdown(f'<div class="badge-ambiguous">⚠ AMBIGUOUS CANDIDATES (Top Score: {match_score:.3f}, Margin: {margin:.3f})</div>', unsafe_allow_html=True)
                st.warning(query_result["message"])
            else:
                st.markdown(f'<div class="badge-unknown">✖ UNKNOWN (Similarity: {match_score:.3f} < {threshold:.2f})</div>', unsafe_allow_html=True)
                st.error(query_result["message"])

            # Metric details (Summary across cameras)
            if top_cand:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Top Match Camera", top_cand["camera_id"])
                m2.metric("Top Track ID", top_cand["track_id"])
                m3.metric("Top Score", f"{match_score:.4f}")
                m4.metric("Cameras Matched", f"{len(active_cameras)} / {len(config.CAMERA_VIDEOS)}")

            # Visual Evidence Generation across ALL cameras
            st.markdown("---")
            st.markdown("### 🎬 Video Evidence with Green Bounding Boxes (All Cameras)")
            st.caption(f"Displaying target tracking evidence across all {len(active_cameras)} cameras with green bounding boxes.")

            if active_cameras and decision != "UNKNOWN":
                gen = get_evidence_generator()
                camera_evidence = {}

                with st.spinner(f"Generating / loading annotated video evidence for all {len(active_cameras)} cameras..."):
                    for cam_id in active_cameras:
                        cand = per_camera_best[cam_id]
                        t_id = cand["track_id"]
                        try:
                            ev = gen.generate_annotated_evidence(
                                track_id=t_id,
                                display_label=f"{cand['identity_label'] if cand['identity_label'] != 'Unregistered' else query_result['identity_label']} ({t_id})",
                                max_frames_to_render=90
                            )
                            camera_evidence[cam_id] = {
                                "candidate": cand,
                                "evidence": ev
                            }
                        except Exception as e:
                            st.error(f"Error loading evidence for {cam_id} ({t_id}): {e}")

                # Create tabs for Multi-Camera Grid plus individual camera inspection
                tab_titles = ["📺 All Cameras (Multi-Camera Grid)"] + [
                    f"{'⭐ ' if cam == top_cand.get('camera_id') else ''}{cam} ({per_camera_best[cam]['score']:.3f})"
                    for cam in active_cameras
                ]
                tabs = st.tabs(tab_titles)

                # Tab 0: All Cameras Side-by-Side (2x2 Grid)
                with tabs[0]:
                    st.markdown("#### Synchronized Multi-Camera Video Evidence Grid")
                    grid_cols = st.columns(2)
                    for idx, cam_id in enumerate(active_cameras):
                        data = camera_evidence.get(cam_id)
                        if not data:
                            continue
                        cand = data["candidate"]
                        ev = data["evidence"]
                        is_top = (top_cand and top_cand.get("camera_id") == cam_id)

                        with grid_cols[idx % 2]:
                            st.markdown(
                                f"**{'⭐ ' if is_top else ''}{cam_id}** — "
                                f"`{cand['track_id']}` | Score: **{cand['score']:.4f}** "
                                f"{'🟢' if cand['score'] >= threshold else '🔴'}"
                            )
                            st.caption(f"⏱ Time: {format_timestamp(cand['start_time'])} → {format_timestamp(cand['end_time'])} ({cand['duration']}s)")

                            ev_path = ev["evidence_video_path"]
                            if os.path.exists(ev_path):
                                with open(ev_path, "rb") as vf:
                                    st.video(vf.read(), format="video/mp4")
                            else:
                                st.error(f"Evidence video file missing for {cam_id}")

                            # Show keyframe thumbnails below video
                            kfs = ev.get("keyframes", [])
                            if kfs:
                                kf_cols = st.columns(min(len(kfs), 3))
                                for k_idx, kf in enumerate(kfs[:3]):
                                    with kf_cols[k_idx]:
                                        if os.path.exists(kf["image_path"]):
                                            st.image(kf["image_path"], caption=f"{format_timestamp(kf['timestamp'])}", use_container_width=True)
                            st.markdown("---")

                # Tabs 1..N: Individual Camera Detailed Views
                for idx, cam_id in enumerate(active_cameras):
                    with tabs[idx + 1]:
                        data = camera_evidence.get(cam_id)
                        if not data:
                            st.warning(f"No evidence data found for {cam_id}")
                            continue
                        cand = data["candidate"]
                        ev = data["evidence"]
                        is_top = (top_cand and top_cand.get("camera_id") == cam_id)

                        st.markdown(f"#### {'⭐ Top Match: ' if is_top else ''}{cam_id} Evidence Inspection")
                        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                        c_m1.metric("Camera ID", cam_id)
                        c_m2.metric("Track ID", cand["track_id"])
                        c_m3.metric("Similarity Score", f"{cand['score']:.4f}")
                        c_m4.metric("Time Observed", f"{format_timestamp(cand['start_time'])} → {format_timestamp(cand['end_time'])} ({cand['duration']}s)")

                        col_v, col_kf = st.columns([3, 2])
                        with col_v:
                            st.markdown(f"**Playable Video Clip ({cam_id}):**")
                            ev_path = ev["evidence_video_path"]
                            if os.path.exists(ev_path):
                                with open(ev_path, "rb") as vf:
                                    st.video(vf.read(), format="video/mp4")
                            else:
                                st.error("Evidence video file could not be opened.")

                        with col_kf:
                            st.markdown(f"**Keyframe Snapshots with Green Bounding Boxes ({cam_id}):**")
                            keyframes = ev.get("keyframes", [])
                            for kf in keyframes[:3]:
                                if os.path.exists(kf["image_path"]):
                                    st.image(kf["image_path"], caption=f"Timestamp: {format_timestamp(kf['timestamp'])}", use_container_width=True)
            else:
                st.info("Visual evidence generation is withheld when match confidence is insufficient or Unknown.")

            # ── Per-Camera Best Match Panel ──────────────────────────────────
            st.markdown("---")
            st.markdown("### 📷 Best Match Per Camera")
            per_camera_best = query_result.get("per_camera_best", {})
            cam_ids = list(config.CAMERA_VIDEOS.keys())
            pcam_cols = st.columns(len(cam_ids))
            for col_idx, cam_id in enumerate(cam_ids):
                with pcam_cols[col_idx]:
                    is_top = (top_cand and top_cand.get("camera_id") == cam_id)
                    header = f"{'⭐ ' if is_top else ''}**{cam_id}**"
                    st.markdown(header)
                    if cam_id in per_camera_best:
                        best = per_camera_best[cam_id]
                        crop_p = best["representative_crop_path"]
                        if os.path.exists(crop_p):
                            st.image(crop_p, use_container_width=True)
                        score_color = "🟢" if best["score"] >= threshold else "🔴"
                        st.caption(f"{score_color} Score: **{best['score']:.4f}**")
                        st.caption(f"Track: `{best['track_id']}`")
                        st.caption(f"Identity: `{best['identity_label']}`")
                        st.caption(f"{format_timestamp(best['start_time'])} → {format_timestamp(best['end_time'])}")
                    else:
                        st.caption("_No tracks found_")

            # ── Top Global Candidates Table ──────────────────────────────────
            st.markdown("---")
            st.markdown("### Candidate Tracks Comparison (Top Hits)")
            candidates = query_result.get("candidates", [])
            display_top_n = min(len(candidates), num_cameras * 2)
            if candidates:
                cand_cols = st.columns(min(display_top_n, 5))
                for i, c in enumerate(candidates[:display_top_n]):
                    with cand_cols[i % 5]:
                        crop_p = c["representative_crop_path"]
                        if os.path.exists(crop_p):
                            st.image(crop_p, use_container_width=True)
                        st.markdown(f"**#{i+1}: {c['track_id']}**")
                        st.caption(f"Camera: `{c['camera_id']}`")
                        st.caption(f"Score: **{c['score']:.4f}**")
                        st.caption(f"Identity: `{c['identity_label']}`")
                        st.caption(f"Time: {format_timestamp(c['start_time'])} - {format_timestamp(c['end_time'])}")


# ==============================================================================
# OPTION B: UPLOAD VIDEO / REGISTER IDENTITIES
# ==============================================================================
elif workflow_mode == "Upload Video / Register Identities":
    st.markdown('<div class="main-header">📋 Upload Video / Register Identities</div>', unsafe_allow_html=True)
    num_cams = len(config.CAMERA_VIDEOS)
    st.markdown(f'<div class="sub-header">Inspect ingested tracks from all {num_cams} cameras, assign identity labels, and manage cross-camera identity mappings in <code>resources.json</code>.</div>', unsafe_allow_html=True)

    # Ingestion Status Banner
    st.markdown("### 1. Ingestion Pipeline Status")
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        st.write(f"• **Database Collection**: `{config.QDRANT_COLLECTION_NAME}` ({total_tracks_db} total points)")
        for cam_id, cam_path in config.CAMERA_VIDEOS.items():
            n = qstore.count_tracks(cam_id)
            st.write(f"• **{cam_id}**: `{cam_path.name}` ({n} tracks in DB)")
    with status_col2:
        if st.button("🔄 Re-Ingest All Videos", help="Reruns detection, tracking, and Qdrant ingestion for all videos"):
            with st.spinner("Processing all videos into Qdrant..."):
                from scripts.ingest_all import main as run_ingest
                run_ingest()
            st.success("Re-ingestion completed!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 2. Representative Track Gallery & Identity Registration")

    selected_cam = st.radio("Choose Camera for Registration:", list(config.CAMERA_VIDEOS.keys()), horizontal=True)

    meta_file = config.METADATA_DIR / f"{selected_cam}_tracks.json"
    if not meta_file.exists():
        st.error(f"Metadata file missing for {selected_cam}. Please run video ingestion first.")
        st.stop()

    tracks_dict = load_json(meta_file, default={})
    st.write(f"Displaying **{len(tracks_dict)}** detected tracks from **{selected_cam}**:")

    # Sort tracks by total_frames_observed descending so clearest, longest tracks appear first
    sorted_tracks = sorted(tracks_dict.values(), key=lambda t: t["total_frames_observed"], reverse=True)

    # Filter options
    filter_choice = st.selectbox("Filter Gallery:", ["All Tracks", "Unregistered Tracks Only", "Registered Tracks Only"])
    display_tracks = []
    for t in sorted_tracks:
        is_reg, assigned_id = registry.is_track_registered(t["track_id"])
        if filter_choice == "Unregistered Tracks Only" and is_reg:
            continue
        if filter_choice == "Registered Tracks Only" and not is_reg:
            continue
        display_tracks.append((t, is_reg, assigned_id))

    st.write(f"Showing **{len(display_tracks)}** matching tracks:")

    # Gallery display in 4-column layout
    num_cols = 4
    for i in range(0, len(display_tracks), num_cols):
        batch = display_tracks[i:i + num_cols]
        cols = st.columns(num_cols)
        for j, (track, is_reg, assigned_id) in enumerate(batch):
            with cols[j]:
                crop_p = track["representative_crop_path"]
                if os.path.exists(crop_p):
                    st.image(crop_p, use_container_width=True)
                
                st.markdown(f"**{track['track_id']}**")
                st.caption(f"⏱ {format_timestamp(track['start_time'])} -> {format_timestamp(track['end_time'])} ({track['duration']}s)")
                st.caption(f"Frames: **{track['total_frames_observed']}**")

                if is_reg:
                    st.success(f"Registered as: **{assigned_id}**")
                else:
                    all_cam_ids = list(config.CAMERA_VIDEOS.keys())
                    primary_cam = all_cam_ids[0]  # Camera_1 is the primary / anchor camera
                    if selected_cam == primary_cam:
                        # Primary camera: manual identity label assignment
                        default_label = registry._next_available_identity_label()
                        user_label = st.text_input(
                            f"Identity Label:",
                            value=default_label,
                            key=f"label_{track['track_id']}"
                        )
                        if st.button("Register Track", key=f"btn_{track['track_id']}"):
                            success, msg = registry.register_camera1_track(user_label, track)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        # All other cameras: auto-match against registry or create new identity
                        if st.button("Match / Register Track", key=f"btn_{track['track_id']}"):
                            res = registry.register_secondary_track(
                                track,
                                match_threshold=config.DEFAULT_MATCH_THRESHOLD,
                                ambiguity_margin=config.AMBIGUITY_THRESHOLD
                            )
                            st.info(res["message"])
                            st.rerun()

    # Section 3: Registered Identity Registry (resources.json)
    st.markdown("---")
    st.markdown("### 3. Persistent Identity Registry (`resources.json`)")

    registry.load()
    if registry.identities:
        for id_label, data in registry.identities.items():
            with st.expander(f"👤 **{id_label}** ({len(data['references'])} references across cameras)", expanded=True):
                ref_cols = st.columns(max(len(data["references"]), 1))
                for idx, ref in enumerate(data["references"]):
                    with ref_cols[idx]:
                        if os.path.exists(ref["crop_path"]):
                            st.image(ref["crop_path"], caption=f"{ref['camera_id']} ({ref['track_id']})", use_container_width=True)
                        st.caption(f"Time: {format_timestamp(ref['start_time'])} - {format_timestamp(ref['end_time'])}")
                        st.caption(f"Vector: {len(ref['vector'])}-dim OSNet")

        col_clr, _ = st.columns([1, 4])
        with col_clr:
            if st.button("🗑 Reset resources.json", help="Clears all registered identities"):
                registry.reset_registry()
                st.success("Registry cleared.")
                st.rerun()
    else:
        st.info("No identities registered in `resources.json` yet. Select tracks from the gallery above to register.")
