"""
app.py — FastAPI backend for the live-webcam+mic pet gesture prototype.

Signals combined into one gesture readout:
  - dog presence:  general COCO-pretrained gate, runs BEFORE anything else (dog_detector.py)
  - emotion:       per-frame CNN classification, gated on confidence + margin (model.py)
  - motion:        pose-based paw velocity if trained (pose_motion.py),
                    else frame-differencing fallback (motion.py)
  - posture:       Sitting/Standing/Lying from pose joint geometry (pose_motion.py)
  - vocalization:  pretrained YAMNet sound-event model, with a freshness check (audio.py)
  - fusion:        weighted cue-matching engine (fusion.py), fed real confidences + pose cues
  - smoothing:     the displayed gesture is the mode of the last few frames, not a
                    single noisy instant
  - alerts:        heuristic welfare flags from pose history (wellness_alerts.py) —
                    only active once the pose model is trained

State is per-session, via Starlette's signed-cookie SessionMiddleware plus a
server-side dict keyed by session id, so multiple simultaneous clients don't
corrupt each other's motion/vocalization/gesture history. This mirrors the
Flask version's `session["sid"]` + `SESSIONS[sid]` pattern; Starlette's
`request.session` is itself a signed cookie (itsdangerous-based), so nothing
about the session-security story changes.

Run:
    pip install fastapi "uvicorn[standard]" itsdangerous python-multipart
    uvicorn app:app --host 0.0.0.0 --port 5000

    (or: python app.py, which calls uvicorn.run() directly -- fine for local
    dev, but --reload and multi-worker mode only work from the `uvicorn`
    CLI, not from this __main__ block.)

Then open http://localhost:5000 and allow webcam + microphone access.
"""

import base64
import inspect
import cv2
import tempfile
import io
import os
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image

from model import load_inference_model, preprocess_frame, passes_confidence_gate, CLASSES
from motion import to_gray_small, classify_motion
from pose_motion import (load_pose_model, extract_keypoints, classify_motion_from_pose,
                          classify_posture, compute_tail_low)
from wellness_alerts import WellnessTracker
from audio import classify_vocalization, decode_pcm16_base64
from fusion import fuse

# YOLO dog filter removed per user request: all frames and uploads are processed directly
_dog_detector_available = False

app = FastAPI()

# A real deployment should pin this via an environment variable so sessions
# survive a server restart; a random key per-process is fine for local dev.
# NOTE: os.urandom(24) is bytes -- SessionMiddleware's secret_key wants a
# str (it's passed straight to itsdangerous), so it's hex-encoded here.
_secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24).hex()
app.add_middleware(SessionMiddleware, secret_key=_secret_key)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

model, is_trained = load_inference_model()
pose_model = load_pose_model()  # None if not trained yet -> falls back to pixel-diff motion

VOCAL_FRESHNESS_SECONDS = 4.0  # a vocalization reading older than this is treated as stale
GESTURE_SMOOTHING_WINDOW = 5   # displayed gesture = mode of the last N raw gestures

# Per-session state, keyed by a signed-cookie session id. Fixes the earlier
# bug where a single module-level `state` dict was shared by every client.
SESSIONS: dict = {}


FOLDER_TO_EMOTION = {
    "angrydogs": "Angry",
    "gooddogs": "Relaxed",
    "sleepydogs": "Sad",
    "smileydogs": "Happy",
}


class PredictRequest(BaseModel):
    image: str = ""
    skip_dog_gate: bool = False


class AudioRequest(BaseModel):
    pcm16: str = ""


def get_session_state(request: Request) -> dict:
    now = time.time()
    # Evict sessions older than 10 minutes (600 seconds)
    expired = [k for k, v in SESSIONS.items() if now - v.get("last_seen", now) > 600]
    for k in expired:
        del SESSIONS[k]

    # request.session is Starlette's signed-cookie session dict -- the
    # direct equivalent of Flask's `session`. Same signed-cookie mechanics,
    # different attribute name.
    if "sid" not in request.session:
        request.session["sid"] = str(uuid.uuid4())
    sid = request.session["sid"]
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "prev_gray": None,
            "prev_kpts": None,
            "last_vocalization": {"label": "Quiet", "confidence": 0.0, "raw_top_class": "",
                                   "available": True, "ts": 0.0},
            "wellness": WellnessTracker(window=20),
            "gesture_history": deque(maxlen=GESTURE_SMOOTHING_WINDOW),
            "last_seen": now,
        }
    SESSIONS[sid]["last_seen"] = now
    return SESSIONS[sid]


_TEMPLATE_RESPONSE_TAKES_REQUEST_FIRST = (
    "request" in list(inspect.signature(templates.TemplateResponse).parameters)[:1]
)


def render_template(request: Request, name: str, context: dict) -> HTMLResponse:
    """TemplateResponse(...) changed its argument order between Starlette
    releases:

      - older releases:  TemplateResponse(name, context)      (context needs "request")
      - newer releases:  TemplateResponse(request, name, context)

    This is decided ONCE at import time by inspecting the installed
    signature, not by calling it and catching TypeError. A try/except
    around the call is unsafe here because TemplateResponse renders the
    template synchronously inside that same call -- a genuine error
    *inside* index.html (e.g. a Jinja global used with the wrong
    arguments) also raises TypeError, and catching it would retry with
    the swapped argument order instead of surfacing the real bug.
    """
    if _TEMPLATE_RESPONSE_TAKES_REQUEST_FIRST:
        return templates.TemplateResponse(request, name, context)
    return templates.TemplateResponse(name, {**context, "request": request})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return render_template(
        request,
        "index.html",
        {
            "is_trained": is_trained,
            "classes": CLASSES,
            "pose_active": pose_model is not None,
        },
    )


@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    state = get_session_state(request)

    data_url = payload.image
    skip_dog_gate = payload.skip_dog_gate
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data_url)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        return JSONResponse({"error": f"could not decode image: {exc}"}, status_code=400)

    frame_rgb = np.array(pil_img)
    frame_bgr = frame_rgb[:, :, ::-1]

    presence = {"present": True, "confidence": 1.0}

    # --- emotion, with confidence + margin gating ---
    batch = preprocess_frame(frame_bgr)
    probs = model.predict(batch, verbose=0)[0]
    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda p: p[1], reverse=True)
    raw_emotion_label, emotion_conf = ranked[0]
    canonical_emotion = FOLDER_TO_EMOTION.get(raw_emotion_label.lower(), raw_emotion_label)
    sorted_probs = [p for _, p in ranked]
    emotion_is_confident = passes_confidence_gate(sorted_probs)

    # --- motion + posture: pose-based if a trained pose model exists, else pixel-diff fallback ---
    posture_result = {"label": "Unknown", "confidence": 0.0}
    curr_kpts, bbox_diag = None, None
    if pose_model is not None:
        pose_res = extract_keypoints(pose_model, frame_bgr)
        curr_kpts, bbox_diag = pose_res[0], pose_res[1]
        motion_result = classify_motion_from_pose(state["prev_kpts"], curr_kpts, bbox_diag)
        if curr_kpts is not None:
            posture_result = classify_posture(curr_kpts, bbox_diag)
        state["prev_kpts"] = curr_kpts
    else:
        curr_gray_small = to_gray_small(frame_bgr)
        motion_result = classify_motion(state["prev_gray"], curr_gray_small)
        state["prev_gray"] = curr_gray_small

    # --- vocalization: only trust it if it's fresh; a stale reading is
    #     effectively "we don't currently know", not "still barking" ---
    last_vocal = state["last_vocalization"]
    vocal_age = time.time() - last_vocal.get("ts", 0.0)
    if vocal_age <= VOCAL_FRESHNESS_SECONDS and last_vocal.get("available", True):
        vocalization_label = last_vocal["label"]
        vocal_conf = last_vocal["confidence"]
    else:
        vocalization_label = "Quiet"
        vocal_conf = None  # unknown, not "confidently quiet" -- don't let it drag scores down

    # --- fuse: real confidences + pose cues, not placeholders ---
    fused_emotion_label = canonical_emotion if emotion_is_confident else "Uncertain"
    pose_cues = {"posture": posture_result["label"],
                 "tail_low": compute_tail_low(curr_kpts, bbox_diag) if curr_kpts is not None else False}
    fused = fuse(fused_emotion_label, emotion_conf, motion_result["label"],
                 vocalization_label, vocal_conf, pose_cues)

    # --- temporal smoothing: displayed gesture is the mode of recent raw gestures ---
    state["gesture_history"].append(fused["gesture"])
    smoothed_gesture = Counter(state["gesture_history"]).most_common(1)[0][0]

    # --- welfare alerts: only meaningful with real keypoints, not the pixel-diff fallback ---
    active_alerts = []
    if pose_model is not None:
        active_alerts = state["wellness"].update(curr_kpts, bbox_diag, motion_result["label"],
                                                   posture_label=posture_result["label"])

    return {
        "dog_present": True,
        "dog_gate_active": _dog_detector_available,
        "dog_confidence": presence["confidence"],
        "is_trained": is_trained,
        "pose_active": pose_model is not None,
        "emotion": {
            "label": canonical_emotion,
            "raw_label": raw_emotion_label,
            "confidence": emotion_conf,
            "confident": emotion_is_confident,
            "all": [{"label": FOLDER_TO_EMOTION.get(l.lower(), l), "confidence": c} for l, c in ranked],
        },
        "motion": motion_result,
        "posture": posture_result,
        "vocalization": {**last_vocal, "fresh": vocal_age <= VOCAL_FRESHNESS_SECONDS},
        "gesture": smoothed_gesture,
        "gesture_raw": fused["gesture"],
        "fusion_score": fused["score"],
        "alerts": active_alerts,
    }


def annotate_pose(bgr_image: np.ndarray, kpts: np.ndarray) -> str:
    """Draws keypoints and skeleton on a copy of the frame, returns base64 data URL."""
    annotated = bgr_image.copy()
    if kpts is not None and len(kpts) > 0:
        connections = [
            ("front_left_paw", "front_left_knee"),
            ("front_left_knee", "front_left_elbow"),
            ("front_left_elbow", "withers"),
            ("front_right_paw", "front_right_knee"),
            ("front_right_knee", "front_right_elbow"),
            ("front_right_elbow", "withers"),
            ("rear_left_paw", "rear_left_knee"),
            ("rear_left_knee", "rear_left_elbow"),
            ("rear_left_elbow", "tail_start"),
            ("rear_right_paw", "rear_right_knee"),
            ("rear_right_knee", "rear_right_elbow"),
            ("rear_right_elbow", "tail_start"),
            ("withers", "throat"),
            ("throat", "chin"),
            ("chin", "nose"),
            ("withers", "tail_start"),
            ("tail_start", "tail_end"),
        ]
        try:
            from pose_motion import KP
            thickness = max(2, int(annotated.shape[1] / 250))
            radius = max(3, int(annotated.shape[1] / 180))
            for p1_name, p2_name in connections:
                if p1_name in KP and p2_name in KP:
                    i1, i2 = KP[p1_name], KP[p2_name]
                    if i1 < len(kpts) and i2 < len(kpts):
                        pt1 = (int(kpts[i1][0]), int(kpts[i1][1]))
                        pt2 = (int(kpts[i2][0]), int(kpts[i2][1]))
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv2.line(annotated, pt1, pt2, (115, 163, 127), thickness)

            for pt in kpts:
                x, y = int(pt[0]), int(pt[1])
                if x > 0 and y > 0:
                    cv2.circle(annotated, (x, y), radius, (61, 163, 232), -1)
        except Exception:
            pass

    max_dim = 960
    h, w = annotated.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def process_single_image(frame_bgr: np.ndarray) -> dict:
    """Core pipeline for single-image analysis without any blocking YOLO dog filters."""
    # --- emotion classification ---
    batch = preprocess_frame(frame_bgr)
    probs = model.predict(batch, verbose=0)[0]
    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda p: p[1], reverse=True)
    raw_emotion_label, emotion_conf = ranked[0]
    canonical_emotion = FOLDER_TO_EMOTION.get(raw_emotion_label.lower(), raw_emotion_label)
    sorted_probs = [p for _, p in ranked]
    emotion_is_confident = passes_confidence_gate(sorted_probs)

    # --- pose / posture ---
    posture_result = {"label": "Unknown", "confidence": 0.0}
    curr_kpts, bbox_diag = None, None
    if pose_model is not None:
        pose_res = extract_keypoints(pose_model, frame_bgr)
        curr_kpts, bbox_diag = pose_res[0], pose_res[1]
        if curr_kpts is not None:
            posture_result = classify_posture(curr_kpts, bbox_diag)

    # For single uploaded image, motion is Still (single frame)
    motion_result = {"label": "Still", "score": 0.0}
    vocalization_label = "Quiet"
    vocal_conf = None

    # Use the canonical emotion directly for fusion
    fused_emotion_label = canonical_emotion
    tail_low_val = bool(compute_tail_low(curr_kpts, bbox_diag)) if curr_kpts is not None else False
    pose_cues = {
        "posture": posture_result["label"],
        "tail_low": tail_low_val,
    }
    fused = fuse(
        fused_emotion_label,
        emotion_conf,
        motion_result["label"],
        vocalization_label,
        vocal_conf,
        pose_cues,
    )

    gesture_name = fused["gesture"]
    # If the rule-based fusion output the generic "Emotion · Still · Quiet", enhance to natural wording
    if " · " in gesture_name:
        posture = posture_result.get("label", "Unknown")
        posture_suffix = f" ({posture.lower()})" if posture != "Unknown" else ""
        descriptive_map = {
            "Happy": f"Content and playful{posture_suffix}",
            "Relaxed": f"Calm and resting{posture_suffix}",
            "Sad": f"Quiet and withdrawn{posture_suffix}",
            "Angry": f"Alert / watchful stance{posture_suffix}",
        }
        gesture_name = descriptive_map.get(canonical_emotion, f"{canonical_emotion} pet state{posture_suffix}")

    annotated_b64 = annotate_pose(frame_bgr, curr_kpts)

    return {
        "success": True,
        "type": "image",
        "dog_present": True,
        "dog_gate_active": False,
        "dog_confidence": 1.0,
        "is_trained": is_trained,
        "pose_active": pose_model is not None,
        "emotion": {
            "label": canonical_emotion,
            "raw_label": raw_emotion_label,
            "confidence": float(emotion_conf),
            "confident": bool(emotion_is_confident),
            "all": [{"label": str(FOLDER_TO_EMOTION.get(l.lower(), l)), "confidence": float(c)} for l, c in ranked],
        },
        "motion": motion_result,
        "posture": posture_result,
        "vocalization": {"label": "N/A (Image)", "confidence": None, "available": False},
        "gesture": gesture_name,
        "gesture_raw": fused["gesture"],
        "fusion_score": float(fused["score"]),
        "annotated_image": annotated_b64,
        "alerts": [],
    }


async def handle_image_upload(target: UploadFile, skip_dog_gate: bool = False):
    """Core handler for uploaded image files."""
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ext = Path(target.filename or "upload.jpg").suffix.lower()
    if ext not in allowed_exts:
        return JSONResponse(
            {"error": f"Unsupported image format ({ext}). Please upload JPG, PNG, WEBP, or BMP."},
            status_code=400,
        )

    try:
        content = await target.read()
        if not content:
            return JSONResponse({"error": "Uploaded image file is empty."}, status_code=400)
        from PIL import ImageOps
        raw_pil = Image.open(io.BytesIO(content)).convert("RGB")
        pil_img = ImageOps.exif_transpose(raw_pil)

        frame_rgb = np.array(pil_img)
        frame_bgr = frame_rgb[:, :, ::-1]

        # Downscale large images (e.g. phone camera 4000x3000) for fast inference
        h, w = frame_bgr.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

        result = process_single_image(frame_bgr)
        return result
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Failed to process image: {exc}"}, status_code=500)


async def handle_video_upload(target: UploadFile):
    """Core handler for uploaded video files without any dog gate filters."""
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    filename = target.filename or "uploaded_video.mp4"
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        return JSONResponse(
            {"error": "Unsupported video format. Use MP4, AVI, MOV, MKV or WEBM."},
            status_code=400,
        )

    temp_path = None
    cap = None

    try:
        data = await target.read()
        if not data:
            return JSONResponse({"error": "Uploaded video is empty."}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Could not open uploaded video."}, status_code=400)

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames else 0.0

        # Sample at up to 5 FPS to keep inference responsive
        sample_fps = min(5.0, fps)
        frame_interval = max(1, int(round(fps / sample_fps)))
        actual_sample_fps = fps / frame_interval

        prev_kpts = None
        prev_gray = None
        gesture_history = deque(maxlen=GESTURE_SMOOTHING_WINDOW)
        frame_results = []
        frame_number = 0
        processed_frames = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_number % frame_interval != 0:
                frame_number += 1
                continue

            processed_frames += 1
            timestamp = round(frame_number / fps, 2)

            # -------- emotion --------
            batch = preprocess_frame(frame_bgr)
            probs = model.predict(batch, verbose=0)[0]
            ranked = sorted(
                zip(CLASSES, probs.tolist()),
                key=lambda p: p[1],
                reverse=True,
            )
            raw_emotion_label, emotion_conf = ranked[0]
            canonical_emotion = FOLDER_TO_EMOTION.get(
                raw_emotion_label.lower(), raw_emotion_label
            )
            emotion_is_confident = passes_confidence_gate(
                [p for _, p in ranked]
            )

            # -------- pose / motion / posture --------
            curr_kpts = None
            bbox_diag = None
            posture_result = {"label": "Unknown", "confidence": 0.0}

            if pose_model is not None:
                pose_res = extract_keypoints(pose_model, frame_bgr)
                curr_kpts, bbox_diag = pose_res[0], pose_res[1]
                motion_result = classify_motion_from_pose(
                    prev_kpts, curr_kpts, bbox_diag
                )
                if curr_kpts is not None:
                    posture_result = classify_posture(curr_kpts, bbox_diag)
                prev_kpts = curr_kpts
            else:
                curr_gray = to_gray_small(frame_bgr)
                motion_result = classify_motion(prev_gray, curr_gray)
                prev_gray = curr_gray

            # -------- vocalization --------
            vocalization_label = "Unknown"
            vocal_conf = None

            # -------- fusion --------
            # Use canonical_emotion directly (don't gate to "Uncertain" for video —
            # the per-frame emotion is real, just use it even if confidence is moderate)
            fused_emotion_label = canonical_emotion
            pose_cues = {
                "posture": posture_result["label"],
                "tail_low": (
                    bool(compute_tail_low(curr_kpts, bbox_diag))
                    if curr_kpts is not None
                    else False
                ),
            }

            fused = fuse(
                fused_emotion_label,
                emotion_conf,
                motion_result["label"],
                vocalization_label,
                vocal_conf,
                pose_cues,
            )

            # Apply same descriptive gesture naming as single-image pipeline
            raw_gesture = fused["gesture"]
            if " · " in raw_gesture:
                posture_lbl = posture_result.get("label", "Unknown")
                posture_suffix = f" ({posture_lbl.lower()})" if posture_lbl != "Unknown" else ""
                _desc_map = {
                    "Happy": f"Content and playful{posture_suffix}",
                    "Relaxed": f"Calm and resting{posture_suffix}",
                    "Sad": f"Quiet and withdrawn{posture_suffix}",
                    "Angry": f"Alert / watchful stance{posture_suffix}",
                }
                raw_gesture = _desc_map.get(canonical_emotion, f"{canonical_emotion} pet state{posture_suffix}")

            gesture_history.append(raw_gesture)
            smoothed_gesture = Counter(gesture_history).most_common(1)[0][0]

            frame_results.append({
                "frame": int(frame_number),
                "time": float(timestamp),
                "dog_present": True,
                "dog_confidence": 1.0,
                "emotion": {
                    "label": str(canonical_emotion),
                    "raw_label": str(raw_emotion_label),
                    "confidence": float(emotion_conf),
                    "confident": bool(emotion_is_confident),
                },
                "motion": {
                    "label": str(motion_result.get("label", "Unknown")),
                    "score": float(motion_result.get("score", 0.0)),
                },
                "posture": {
                    "label": str(posture_result.get("label", "Unknown")),
                    "confidence": float(posture_result.get("confidence", 0.0)),
                },
                "tail_low": bool(pose_cues["tail_low"]),
                "vocalization": {
                    "label": str(vocalization_label),
                    "confidence": vocal_conf,
                    "available": False,
                },
                "gesture_raw": str(fused["gesture"]),
                "gesture": str(smoothed_gesture),
                "fusion_score": float(fused["score"]),
            })

            frame_number += 1

        valid = frame_results

        if valid:
            gesture_counts = Counter(r["gesture"] for r in valid)
            emotion_counts = Counter(r["emotion"]["label"] for r in valid)
            motion_counts = Counter(r["motion"]["label"] for r in valid)
            posture_counts = Counter(r["posture"]["label"] for r in valid)

            final_gesture = gesture_counts.most_common(1)[0][0]
            final_emotion = emotion_counts.most_common(1)[0][0]
            final_motion = motion_counts.most_common(1)[0][0]
            final_posture = posture_counts.most_common(1)[0][0]
        else:
            gesture_counts = Counter()
            emotion_counts = Counter()
            motion_counts = Counter()
            posture_counts = Counter()
            final_gesture = "No movement detected"
            final_emotion = "Unknown"
            final_motion = "Unknown"
            final_posture = "Unknown"

        return {
            "success": True,
            "type": "video",
            "video": {
                "filename": filename,
                "fps": round(fps, 2),
                "duration_seconds": round(duration, 2),
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "sample_fps": round(actual_sample_fps, 2),
            },
            "models": {
                "emotion_active": True,
                "pose_active": pose_model is not None,
                "dog_gate_active": False,
            },
            "final_fusion": {
                "gesture": final_gesture,
                "emotion": final_emotion,
                "motion": final_motion,
                "posture": final_posture,
                "audio": "Not processed in video endpoint",
            },
            "gesture_counts": dict(gesture_counts),
            "emotion_counts": dict(emotion_counts),
            "motion_counts": dict(motion_counts),
            "posture_counts": dict(posture_counts),
            "frames": frame_results,
        }

    except Exception as exc:
        return JSONResponse(
            {"error": f"Video processing failed: {exc}"},
            status_code=500,
        )
    finally:
        if cap is not None:
            cap.release()
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/predict_image")
async def predict_image(
    file: UploadFile = File(None),
    image: UploadFile = File(None),
):
    """Run gesture recognition on an uploaded pet image."""
    target = file if isinstance(file, UploadFile) else (image if isinstance(image, UploadFile) else None)
    if not target:
        return JSONResponse({"error": "No image file provided."}, status_code=400)
    return await handle_image_upload(target)


@app.post("/predict_video")
async def predict_video(
    video: UploadFile = File(None),
    file: UploadFile = File(None),
):
    """Run the visual fusion pipeline on an uploaded video."""
    target = video if isinstance(video, UploadFile) else (file if isinstance(file, UploadFile) else None)
    if not target:
        return JSONResponse({"error": "No video file provided."}, status_code=400)
    return await handle_video_upload(target)


@app.post("/upload")
async def upload_media(
    file: UploadFile = File(...),
):
    """Unified endpoint to analyze an uploaded image or video."""
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    ct = (file.content_type or "").lower()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if ext in image_exts or ct.startswith("image/"):
        return await handle_image_upload(file)
    elif ext in video_exts or ct.startswith("video/"):
        return await handle_video_upload(file)
    else:
        return JSONResponse(
            {"error": f"Unsupported file format '{ext or ct}'. Please upload an image (JPG, PNG, WEBP) or video (MP4, MOV, WEBM)."},
            status_code=400,
        )


@app.post("/predict_audio")
def predict_audio(payload: AudioRequest, request: Request):
    state = get_session_state(request)

    b64 = payload.pcm16
    if not b64:
        return JSONResponse({"error": "empty audio payload -- no PCM data received"}, status_code=400)

    try:
        raw_bytes = base64.b64decode(b64)
        waveform = decode_pcm16_base64(raw_bytes)
    except Exception as exc:
        return JSONResponse({"error": f"could not decode audio: {exc}"}, status_code=400)

    if waveform.size < 1600:  # need at least ~0.1s at 16kHz
        return JSONResponse(
            {"error": "audio buffer too short (mic may not be delivering data)"}, status_code=400)

    result = classify_vocalization(waveform)
    result["ts"] = time.time()
    state["last_vocalization"] = result
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)