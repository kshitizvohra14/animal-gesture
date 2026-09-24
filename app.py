"""
app.py — FastAPI backend for the live-webcam+mic pet gesture prototype.

Signals combined into one gesture readout:
  - dog presence:  strict COCO-pretrained gate, runs BEFORE emotion/pose inference (dog_detector.py)
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
import subprocess
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
                          classify_posture, compute_tail_low, PoseStateTracker)
from wellness_alerts import WellnessTracker
from audio import classify_vocalization, decode_pcm16_base64, get_status as get_audio_status
from fusion import fuse, resolve_final_emotion
from dog_detector import load_dog_detector, detect_dog, crop_dog

# IMPORTANT: the dog gate runs BEFORE the dog-emotion model.
# If the detector cannot load, we fail closed instead of treating humans as dogs.
_dog_detector = load_dog_detector()
_dog_detector_available = _dog_detector is not None

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
            "pose_tracker": PoseStateTracker(),
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



def pose_features_to_cues(pose_features: dict) -> dict:
    """Convert PoseStateTracker output into the format expected by fusion.py."""
    posture = pose_features.get("posture", {})
    tail = pose_features.get("tail", {})
    ears = pose_features.get("ears", {})
    head = pose_features.get("head", {})
    paw = pose_features.get("paw", {})

    return {
        "posture": posture.get("label", "Unknown"),
        "posture_confidence": float(posture.get("confidence", 0.0)),
        "tail": tail,
        "ears": ears,
        "head": head,
        "paw": paw,
        "tail_low": bool(tail.get("low", False)),
        "tail_raised": bool(tail.get("raised", False)),
        "tail_moving": bool(tail.get("moving", False)),
        "tail_wagging": {
            "active": bool(tail.get("wagging", False)),
            "confidence": float(tail.get("wag_score", 0.0)),
        },
        "paw_raised": paw.get("label") not in {None, "None", ""},
    }

def dog_gate_response(presence: dict, media_type: str = "image") -> dict:
    """Standard response when no sufficiently confident dog is detected."""
    if not presence.get("available", False):
        message = (
            "Dog detector is unavailable. Install/download yolo11n.pt or set "
            "DOG_DETECTOR_WEIGHTS to a valid YOLO detector file."
        )
    else:
        message = "No dog detected with sufficient confidence."

    return {
        "success": True,
        "type": media_type,
        "dog_present": False,
        "dog_gate_active": bool(presence.get("available", False)),
        "dog_confidence": float(presence.get("confidence", 0.0)),
        "message": message,
        "is_trained": is_trained,
        "pose_active": pose_model is not None,
        "emotion": None,
        "motion": {"label": "Unknown", "score": 0.0},
        "posture": {"label": "Unknown", "confidence": 0.0},
        "vocalization": {"label": "N/A", "confidence": None, "available": False},
        "gesture": "No dog detected",
        "gesture_raw": "No dog detected",
        "fusion_score": 0.0,
        "alerts": [],
    }


def get_dog_crop(frame_bgr: np.ndarray):
    """
    Detect the dog on the full frame, then crop the detected dog before
    emotion/pose inference. This prevents the dog-emotion classifier from
    seeing a human/background and being forced to choose a dog emotion.
    """
    presence = detect_dog(frame_bgr)
    if not presence["present"]:
        return None, presence

    dog_crop, origin = crop_dog(frame_bgr, presence)
    if dog_crop is None or dog_crop.size == 0:
        presence["present"] = False
        presence["bbox"] = None
        return None, presence

    presence["crop_origin"] = origin
    return dog_crop, presence

@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    state = get_session_state(request)

    data_url = payload.image
    skip_dog_gate = bool(payload.skip_dog_gate)
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data_url)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        return JSONResponse({"error": f"could not decode image: {exc}"}, status_code=400)

    frame_rgb = np.array(pil_img)
    frame_bgr = frame_rgb[:, :, ::-1]

    # --- HARD DOG GATE: humans never reach the dog-emotion classifier ---
    if not skip_dog_gate:
        dog_frame, presence = get_dog_crop(frame_bgr)
        if dog_frame is None:
            return dog_gate_response(presence, "image")
        frame_for_model = dog_frame
    else:
        # Only useful for controlled debugging; UI/API should normally leave this false.
        presence = {"present": True, "confidence": 1.0, "available": False}
        frame_for_model = frame_bgr

    # --- emotion, with confidence + margin gating ---
    batch = preprocess_frame(frame_for_model)
    probs = model.predict(batch, verbose=0)[0]
    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda p: p[1], reverse=True)
    raw_emotion_label, emotion_conf = ranked[0]
    canonical_emotion = FOLDER_TO_EMOTION.get(raw_emotion_label.lower(), raw_emotion_label)
    sorted_probs = [p for _, p in ranked]
    emotion_is_confident = passes_confidence_gate(sorted_probs)

    # --- motion + posture + temporal pose behavior cues ---
    posture_result = {"label": "Unknown", "confidence": 0.0}
    curr_kpts, bbox_diag = None, None
    pose_features = PoseStateTracker.empty_features()

    if pose_model is not None:
        pose_res = extract_keypoints(pose_model, frame_for_model)
        curr_kpts, bbox_diag = pose_res[0], pose_res[1]
        pose_features = state["pose_tracker"].update(curr_kpts, bbox_diag)
        motion_result = pose_features["motion"]
        posture_result = pose_features["posture"]
        state["prev_kpts"] = curr_kpts
    else:
        state["pose_tracker"].reset()
        curr_gray_small = to_gray_small(frame_for_model)
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

    # --- fuse all available pose cues ---
    fused_emotion_label = canonical_emotion if emotion_is_confident else "Uncertain"
    pose_cues = pose_features_to_cues(pose_features)

    fused = fuse(fused_emotion_label, emotion_conf, motion_result["label"],
                 vocalization_label, vocal_conf, pose_cues)

    # --- correct the displayed emotion using pose/behavior counter-evidence
    #     (a wagging tail + standing/walking dog cannot plausibly be Sad) ---
    display_label, display_conf, overridden, override_reason = resolve_final_emotion(
        canonical_emotion, emotion_conf, emotion_is_confident, fused,
        motion_result["label"], posture_result["label"],
    )

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
        "dog_gate_active": bool(_dog_detector_available and not skip_dog_gate),
        "dog_confidence": presence["confidence"],
        "is_trained": is_trained,
        "pose_active": pose_model is not None,
        "emotion": {
            "label": display_label,
            "raw_label": raw_emotion_label,
            "confidence": display_conf,
            "confident": emotion_is_confident,
            "overridden": overridden,
            "override_reason": override_reason,
            "all": [{"label": FOLDER_TO_EMOTION.get(l.lower(), l), "confidence": c} for l, c in ranked],
        },
        "motion": motion_result,
        "posture": posture_result,
        "tail": pose_features.get("tail", {}),
        "ears": pose_features.get("ears", {}),
        "head": pose_features.get("head", {}),
        "paw": pose_features.get("paw", {}),
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
    """Analyze an image only after a confident dog detection."""
    dog_frame, presence = get_dog_crop(frame_bgr)

    if dog_frame is None:
        return dog_gate_response(presence, "image")

    # --- emotion classification on DOG CROP only ---
    batch = preprocess_frame(dog_frame)
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
    emotion_is_confident = passes_confidence_gate([p for _, p in ranked])

    # --- pose / posture / behavior cues on DOG CROP ONLY ---
    posture_result = {"label": "Unknown", "confidence": 0.0}
    curr_kpts, bbox_diag = None, None
    image_pose_tracker = PoseStateTracker()
    pose_features = PoseStateTracker.empty_features()

    if pose_model is not None:
        pose_res = extract_keypoints(pose_model, dog_frame)
        curr_kpts, bbox_diag = pose_res[0], pose_res[1]
        pose_features = image_pose_tracker.update(curr_kpts, bbox_diag)
        posture_result = pose_features["posture"]

    # A single uploaded image has no temporal motion information.
    motion_result = {"label": "Still", "score": 0.0, "confidence": 0.0}
    vocalization_label = "Quiet"
    vocal_conf = None

    fused_emotion_label = (
        canonical_emotion if emotion_is_confident else "Uncertain"
    )
    pose_cues = pose_features_to_cues(pose_features)

    fused = fuse(
        fused_emotion_label,
        emotion_conf,
        motion_result["label"],
        vocalization_label,
        vocal_conf,
        pose_cues,
    )

    # --- correct the displayed emotion using pose/behavior counter-evidence
    #     (a wagging tail + standing dog cannot plausibly be Sad) ---
    display_label, display_conf, overridden, override_reason = resolve_final_emotion(
        canonical_emotion, emotion_conf, emotion_is_confident, fused,
        motion_result["label"], posture_result["label"],
    )

    gesture_name = fused["gesture"]
    if not fused["matched_rule"]:
        posture = posture_result.get("label", "Unknown")
        posture_suffix = (
            f" ({posture.lower()})" if posture != "Unknown" else ""
        )
        descriptive_map = {
            "Happy": f"Content and playful{posture_suffix}",
            "Relaxed": f"Calm and resting{posture_suffix}",
            "Sad": f"Quiet and withdrawn{posture_suffix}",
            "Angry": f"Alert / watchful stance{posture_suffix}",
            "Uncertain": f"Hard to read from this frame{posture_suffix}",
        }
        gesture_name = descriptive_map.get(
            display_label,
            f"{display_label} pet state{posture_suffix}",
        )

    annotated_b64 = annotate_pose(dog_frame, curr_kpts)

    return {
        "success": True,
        "type": "image",
        "dog_present": True,
        "dog_gate_active": True,
        "dog_confidence": float(presence["confidence"]),
        "dog_bbox": presence.get("bbox"),
        "is_trained": is_trained,
        "pose_active": pose_model is not None,
        "emotion": {
            "label": display_label,
            "raw_label": raw_emotion_label,
            "confidence": float(display_conf),
            "confident": bool(emotion_is_confident),
            "overridden": overridden,
            "override_reason": override_reason,
            "all": [
                {
                    "label": str(FOLDER_TO_EMOTION.get(l.lower(), l)),
                    "confidence": float(c),
                }
                for l, c in ranked
            ],
        },
        "motion": motion_result,
        "posture": posture_result,
        "tail": pose_features.get("tail", {}),
        "ears": pose_features.get("ears", {}),
        "head": pose_features.get("head", {}),
        "paw": pose_features.get("paw", {}),
        "vocalization": {
            "label": "N/A (Image)",
            "confidence": None,
            "available": False,
        },
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



VIDEO_AUDIO_SAMPLE_RATE = 16000
VIDEO_AUDIO_CHANNELS = 1
VIDEO_AUDIO_WINDOW_SECONDS = 1.0


def get_ffmpeg_executable() -> str:
    """Find a usable ffmpeg executable, preferring bundled imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg") or "ffmpeg"


def extract_video_audio_pcm(video_path: str):
    """
    Extract the uploaded video's audio as mono 16-kHz signed PCM.

    Uses ffmpeg because OpenCV does not decode the audio stream. The returned
    NumPy array is suitable for the existing YAMNet classify_vocalization()
    function in audio.py.
    """
    ffmpeg_bin = get_ffmpeg_executable()
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-ac", str(VIDEO_AUDIO_CHANNELS),
        "-ar", str(VIDEO_AUDIO_SAMPLE_RATE),
        "-f", "s16le",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return None, "ffmpeg is not installed or is not on PATH."

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        return None, f"Could not extract video audio: {err[-500:]}"

    if not proc.stdout:
        return np.empty(0, dtype=np.float32), None

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        return np.empty(0, dtype=np.float32), None

    waveform = pcm.astype(np.float32) / 32768.0
    return waveform, None


def decode_audio_file(audio_path: str):
    """
    Decode an audio file (.mp3, .wav, .m4a, .ogg, .flac, etc.) to 16-kHz mono float32.
    Tries librosa first, then ffmpeg.
    """
    try:
        import librosa
        waveform, _ = librosa.load(audio_path, sr=VIDEO_AUDIO_SAMPLE_RATE, mono=True)
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = np.clip(waveform, -1.0, 1.0)
        return waveform, None
    except Exception as librosa_err:
        ffmpeg_bin = get_ffmpeg_executable()
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-i", audio_path,
            "-vn",
            "-ac", str(VIDEO_AUDIO_CHANNELS),
            "-ar", str(VIDEO_AUDIO_SAMPLE_RATE),
            "-f", "s16le",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode == 0 and proc.stdout:
                pcm = np.frombuffer(proc.stdout, dtype=np.int16)
                if pcm.size > 0:
                    waveform = pcm.astype(np.float32) / 32768.0
                    return waveform, None
        except Exception:
            pass
        return None, f"Could not decode audio: {librosa_err}"


def classify_video_audio_at_time(waveform, timestamp_seconds: float):
    """
    Classify approximately one second of audio around a video timestamp.

    Returns a YAMNet-compatible result dictionary. If the video has no audio,
    the result is explicitly marked unavailable instead of pretending that
    silence was detected.
    """
    if waveform is None:
        return {
            "label": "Audio unavailable",
            "confidence": None,
            "available": False,
            "raw_top_class": "",
        }

    if waveform.size == 0:
        return {
            "label": "No audio track",
            "confidence": None,
            "available": False,
            "raw_top_class": "",
        }

    sr = VIDEO_AUDIO_SAMPLE_RATE
    half = VIDEO_AUDIO_WINDOW_SECONDS / 2.0
    start = max(0, int((timestamp_seconds - half) * sr))
    end = min(waveform.size, int((timestamp_seconds + half) * sr))

    chunk = waveform[start:end]

    # YAMNet works better with enough audio context. If the timestamp is near
    # the beginning/end, pad the chunk with zeros to about one second.
    target_len = int(VIDEO_AUDIO_WINDOW_SECONDS * sr)
    if chunk.size < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        offset = max(0, (target_len - chunk.size) // 2)
        padded[offset:offset + chunk.size] = chunk
        chunk = padded

    try:
        result = classify_vocalization(chunk)
        if not isinstance(result, dict):
            return {
                "label": "Unknown",
                "confidence": None,
                "available": True,
                "raw_top_class": "",
            }

        # BUG: this used to hard-code "available": True regardless of what
        # classify_vocalization() actually reported. When YAMNet failed to
        # load (no network access to tfhub.dev, missing tensorflow_hub,
        # etc.), classify_vocalization() correctly returns
        # {"label": "Quiet", "available": False, ...} -- but this wrapper
        # was silently overwriting that to "available": True, so the UI
        # displayed "Quiet" as if it had genuinely listened and heard
        # nothing, permanently masking the real failure. Propagate the
        # actual availability instead.
        available = bool(result.get("available", False))
        return {
            "label": str(result.get("label", "Unknown")) if available else "Audio unavailable",
            "confidence": (
                float(result["confidence"])
                if available and result.get("confidence") is not None else None
            ),
            "available": available,
            "raw_top_class": str(result.get("raw_top_class", "")),
        }
    except Exception as exc:
        return {
            "label": "Audio error",
            "confidence": None,
            "available": False,
            "raw_top_class": "",
            "error": str(exc),
        }


async def handle_video_upload(target: UploadFile):
    """
    Analyze an uploaded video as a fully multimodal sample.

    Visual stream:
      dog gate -> emotion -> pose/motion/posture -> tail/ears/head/paw

    Audio stream:
      video audio -> ffmpeg -> 16-kHz PCM -> YAMNet -> vocalization

    Both streams are aligned by video timestamp and passed into the same
    fusion engine for every sampled frame.
    """
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

        # Extract audio once. OpenCV handles video frames; ffmpeg handles the
        # separate audio stream.
        audio_waveform, audio_error = extract_video_audio_pcm(temp_path)
        audio_available = audio_waveform is not None and audio_waveform.size > 0

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return JSONResponse(
                {"error": "Could not open uploaded video."},
                status_code=400,
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames else 0.0

        sample_fps = min(5.0, fps)
        frame_interval = max(1, int(round(fps / sample_fps)))
        actual_sample_fps = fps / frame_interval

        prev_gray = None
        pose_tracker = PoseStateTracker()
        gesture_history = deque(maxlen=GESTURE_SMOOTHING_WINDOW)

        frame_results = []
        frame_number = 0
        processed_frames = 0
        dog_frames = 0

        audio_counts = Counter()
        vocal_confidences = []
        audio_events = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_number % frame_interval != 0:
                frame_number += 1
                continue

            processed_frames += 1
            timestamp = round(frame_number / fps, 2)

            # ---------- HARD DOG GATE ----------
            dog_frame, presence = get_dog_crop(frame_bgr)

            if dog_frame is None:
                pose_tracker.reset()
                prev_gray = None
                gesture_history.clear()

                # Audio is classified independently of visual dog detection,
                # and (unlike before) IS counted into the running totals here
                # too -- a bark on a frame where the dog isn't visually
                # detected (motion blur, bad angle, cropped out) is still a
                # real bark and must not be silently dropped from
                # audio_counts/vocal_confidences/audio_events.
                audio_result = classify_video_audio_at_time(
                    audio_waveform, timestamp
                )
                vocalization_label = audio_result["label"]
                vocal_conf = audio_result["confidence"]

                if audio_result.get("available"):
                    audio_counts[vocalization_label] += 1
                    if vocal_conf is not None:
                        vocal_confidences.append(vocal_conf)
                    if vocalization_label not in {
                        "Quiet", "Unknown", "Audio unavailable",
                        "No audio track", "Audio error"
                    }:
                        audio_events += 1

                frame_results.append({
                    "frame": int(frame_number),
                    "time": float(timestamp),
                    "dog_present": False,
                    "dog_confidence": float(presence.get("confidence", 0.0)),
                    "emotion": None,
                    "motion": {"label": "Unknown", "score": 0.0, "confidence": 0.0},
                    "posture": {"label": "Unknown", "confidence": 0.0},
                    "tail": {},
                    "ears": {},
                    "head": {},
                    "paw": {},
                    "vocalization": audio_result,
                    "gesture_raw": "No dog detected",
                    "gesture": "No dog detected",
                    "fusion_score": 0.0,
                })

                frame_number += 1
                continue

            dog_frames += 1

            # ---------- EMOTION ----------
            batch = preprocess_frame(dog_frame)
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

            # ---------- POSE / MOTION / POSTURE ----------
            curr_kpts = None
            bbox_diag = None
            pose_features = PoseStateTracker.empty_features()

            if pose_model is not None:
                pose_res = extract_keypoints(pose_model, dog_frame)
                curr_kpts, bbox_diag = pose_res[0], pose_res[1]
                pose_features = pose_tracker.update(curr_kpts, bbox_diag)
                motion_result = pose_features["motion"]
                posture_result = pose_features["posture"]
            else:
                pose_tracker.reset()
                curr_gray = to_gray_small(dog_frame)
                motion_result = classify_motion(prev_gray, curr_gray)
                prev_gray = curr_gray
                posture_result = {"label": "Unknown", "confidence": 0.0}

            # ---------- AUDIO AT SAME VIDEO TIMESTAMP ----------
            audio_result = classify_video_audio_at_time(
                audio_waveform, timestamp
            )
            vocalization_label = audio_result["label"]
            vocal_conf = audio_result["confidence"]

            if audio_result.get("available"):
                audio_counts[vocalization_label] += 1
                if vocal_conf is not None:
                    vocal_confidences.append(vocal_conf)

                if vocalization_label not in {
                    "Quiet", "Unknown", "Audio unavailable",
                    "No audio track", "Audio error"
                }:
                    audio_events += 1

            # ---------- MULTIMODAL FUSION ----------
            fused_emotion_label = (
                canonical_emotion if emotion_is_confident else "Uncertain"
            )

            pose_cues = pose_features_to_cues(pose_features)

            fused = fuse(
                fused_emotion_label,
                emotion_conf,
                motion_result["label"],
                vocalization_label,
                vocal_conf,
                pose_cues,
            )

            # --- correct the displayed emotion using pose/behavior
            #     counter-evidence (wagging tail + active posture/motion
            #     cannot plausibly be Sad) ---
            display_label, display_conf, overridden, override_reason = resolve_final_emotion(
                canonical_emotion, emotion_conf, emotion_is_confident, fused,
                motion_result["label"], posture_result["label"],
            )

            raw_gesture = fused["gesture"]

            if not fused["matched_rule"]:
                posture_lbl = posture_result.get("label", "Unknown")
                posture_suffix = (
                    f" ({posture_lbl.lower()})"
                    if posture_lbl != "Unknown" else ""
                )

                desc_map = {
                    "Happy": f"Content and playful{posture_suffix}",
                    "Relaxed": f"Calm and resting{posture_suffix}",
                    "Sad": f"Quiet and withdrawn{posture_suffix}",
                    "Angry": f"Alert / watchful stance{posture_suffix}",
                    "Uncertain": f"Hard to read from this frame{posture_suffix}",
                }

                raw_gesture = desc_map.get(
                    display_label,
                    f"{display_label} pet state{posture_suffix}",
                )

            gesture_history.append(raw_gesture)
            smoothed_gesture = Counter(
                gesture_history
            ).most_common(1)[0][0]

            frame_results.append({
                "frame": int(frame_number),
                "time": float(timestamp),
                "dog_present": True,
                "dog_confidence": float(presence["confidence"]),
                "dog_bbox": presence.get("bbox"),

                "emotion": {
                    "label": str(display_label),
                    "raw_label": str(raw_emotion_label),
                    "confidence": float(display_conf),
                    "confident": bool(emotion_is_confident),
                    "overridden": overridden,
                    "override_reason": override_reason,
                },

                "motion": {
                    "label": str(motion_result.get("label", "Unknown")),
                    "score": float(motion_result.get("score", 0.0)),
                    "confidence": float(
                        motion_result.get("confidence", 0.0)
                    ),
                },

                "posture": {
                    "label": str(posture_result.get("label", "Unknown")),
                    "confidence": float(
                        posture_result.get("confidence", 0.0)
                    ),
                },

                "tail": pose_features.get("tail", {}),
                "ears": pose_features.get("ears", {}),
                "head": pose_features.get("head", {}),
                "paw": pose_features.get("paw", {}),

                "vocalization": audio_result,

                "gesture_raw": str(fused["gesture"]),
                "gesture": str(smoothed_gesture),
                "fusion_score": float(fused["score"]),
            })

            frame_number += 1

        valid = [r for r in frame_results if r["dog_present"]]

        # Audio summary uses the GLOBAL audio_counts (every sampled frame,
        # dog visually detected or not) -- not gated by the visual dog gate.
        # A bark is real even on a frame where the dog wasn't visually
        # detected (motion blur, bad angle, cropped out of frame), and
        # gating the audio summary on dog_present silently dropped those
        # here before.
        final_audio = (
            audio_counts.most_common(1)[0][0]
            if audio_counts else
            ("No audio track" if not audio_available else "Unknown")
        )

        if valid:
            gesture_counts = Counter(r["gesture"] for r in valid)
            emotion_counts = Counter(r["emotion"]["label"] for r in valid)
            motion_counts = Counter(r["motion"]["label"] for r in valid)
            posture_counts = Counter(r["posture"]["label"] for r in valid)
            dog_audio_counts = Counter(
                r["vocalization"]["label"]
                for r in valid
                if r["vocalization"].get("available")
            )

            final_gesture = gesture_counts.most_common(1)[0][0]
            final_emotion = emotion_counts.most_common(1)[0][0]
            final_motion = motion_counts.most_common(1)[0][0]
            final_posture = posture_counts.most_common(1)[0][0]
        else:
            gesture_counts = Counter()
            emotion_counts = Counter()
            motion_counts = Counter()
            posture_counts = Counter()
            dog_audio_counts = Counter()

            final_gesture = "No dog detected"
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
                "dog_frames": dog_frames,
                "dog_frame_ratio": (
                    round(dog_frames / processed_frames, 3)
                    if processed_frames else 0.0
                ),
                "sample_fps": round(actual_sample_fps, 2),
            },

            "models": {
                "emotion_active": True,
                "pose_active": pose_model is not None,
                "dog_gate_active": _dog_detector_available,
                "audio_active": audio_available,
                "audio_backend": "ffmpeg + YAMNet",
            },

            "audio": {
                "available": audio_available,
                "error": audio_error,
                "sample_rate": VIDEO_AUDIO_SAMPLE_RATE,
                "window_seconds": VIDEO_AUDIO_WINDOW_SECONDS,
                "event_frames": audio_events,
                "vocalization_counts": dict(audio_counts),
                "mean_confidence": (
                    round(float(np.mean(vocal_confidences)), 3)
                    if vocal_confidences else None
                ),
            },

            "final_fusion": {
                "gesture": final_gesture,
                "emotion": final_emotion,
                "motion": final_motion,
                "posture": final_posture,
                "audio": final_audio,
                "modalities": {
                    "vision": True,
                    "pose": pose_model is not None,
                    "audio": audio_available,
                },
            },

            "gesture_counts": dict(gesture_counts),
            "emotion_counts": dict(emotion_counts),
            "motion_counts": dict(motion_counts),
            "posture_counts": dict(posture_counts),
            "vocalization_counts": dict(dog_audio_counts),

            "tail_wagging_frames": int(
                sum(
                    1 for r in valid
                    if r.get("tail", {}).get("wagging")
                )
            ),
            "tail_moving_frames": int(
                sum(
                    1 for r in valid
                    if r.get("tail", {}).get("moving")
                )
            ),

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


async def handle_audio_upload(target: UploadFile):
    """Core handler for uploaded standalone audio files (.mp3, .wav, .m4a, etc.)."""
    allowed_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
    filename = target.filename or "audio_upload.mp3"
    ext = Path(filename).suffix.lower()
    ct = (target.content_type or "").lower()

    if ext not in allowed_exts and not ct.startswith("audio/"):
        return JSONResponse(
            {"error": f"Unsupported audio format ({ext}). Supported formats: MP3, WAV, M4A, OGG, FLAC, AAC."},
            status_code=400,
        )

    temp_path = None
    try:
        content = await target.read()
        if not content:
            return JSONResponse({"error": "Uploaded audio file is empty."}, status_code=400)

        with tempfile.NamedTemporaryFile(suffix=ext or ".mp3", delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        waveform, err = decode_audio_file(temp_path)
        if waveform is None or waveform.size == 0:
            return JSONResponse({"error": f"Could not decode audio: {err or 'no audio samples detected'}"}, status_code=400)

        duration = float(waveform.size) / VIDEO_AUDIO_SAMPLE_RATE
        result = classify_vocalization(waveform)
        vocal_label = result.get("label", "Quiet")
        confidence = float(result.get("confidence", 0.0))

        vocal_behavior_map = {
            "Barking": ("Alert / Playful", "Barking detected - dog may be excited, alert, or calling for attention."),
            "Growling": ("Angry / Guarding", "Growling detected - dog is displaying defensive or agitated behavior."),
            "Whimpering": ("Sad / Whimpering", "Whimpering detected - dog may be in distress, anxious, or seeking comfort."),
            "Howling": ("Sad / Howling", "Howling detected - dog may be calling, vocalizing, or feeling lonely."),
            "Quiet": ("Relaxed / Quiet", "Quiet / no prominent canine vocalization detected in this clip."),
        }
        display_emotion, desc = vocal_behavior_map.get(
            vocal_label,
            ("Uncertain", f"Canine vocalization: {vocal_label}")
        )

        all_classes = ["Barking", "Whimpering", "Growling", "Howling", "Quiet"]
        vocal_bars = []
        for cls in all_classes:
            if cls == vocal_label:
                vocal_bars.append({"label": cls, "confidence": round(confidence, 3)})
            else:
                rem = max(0.01, round((1.0 - confidence) / (len(all_classes) - 1), 3))
                vocal_bars.append({"label": cls, "confidence": rem})
        vocal_bars.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "success": True,
            "type": "audio",
            "filename": filename,
            "duration": round(duration, 2),
            "dog_present": True,
            "dog_gate_active": False,
            "dog_confidence": round(max(0.6, confidence), 2) if vocal_label != "Quiet" else 0.5,
            "is_trained": is_trained,
            "pose_active": pose_model is not None,
            "emotion": {
                "label": display_emotion,
                "raw_label": vocal_label,
                "confidence": round(confidence, 3),
                "confident": bool(confidence >= 0.25),
                "overridden": True,
                "override_reason": desc,
                "all": vocal_bars,
            },
            "motion": {"label": "Audio only (No video)", "score": 0.0, "confidence": 0.0},
            "posture": {"label": "Audio only (No video)", "confidence": 0.0},
            "vocalization": {
                "label": vocal_label,
                "confidence": round(confidence, 3),
                "raw_top_class": result.get("raw_top_class", ""),
                "available": bool(result.get("available", True)),
                "fresh": True,
            },
            "gesture": f"Vocalization: {vocal_label}",
            "gesture_raw": vocal_label,
            "fusion_score": round(confidence, 2),
            "alerts": [
                {
                    "severity": "critical" if vocal_label == "Growling" else ("watch" if vocal_label in ["Whimpering", "Howling"] else "info"),
                    "message": desc,
                }
            ] if vocal_label != "Quiet" else [],
            "audio": {
                "duration_seconds": round(duration, 2),
                "sample_rate": VIDEO_AUDIO_SAMPLE_RATE,
                "label": vocal_label,
                "confidence": round(confidence, 3),
                "raw_top_class": result.get("raw_top_class", ""),
            }
        }
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": f"Failed to process audio: {exc}"}, status_code=500)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _is_valid_upload(target) -> bool:
    return target is not None and bool(getattr(target, "filename", None))


@app.post("/predict_audio_file")
async def predict_audio_file(
    file: UploadFile = File(None),
    audio: UploadFile = File(None),
):
    """Run vocalization detection on an uploaded audio file (.mp3, .wav, etc.)."""
    target = file if _is_valid_upload(file) else (audio if _is_valid_upload(audio) else None)
    if not target:
        return JSONResponse({"error": "No audio file provided."}, status_code=400)
    return await handle_audio_upload(target)


@app.post("/predict_image")
async def predict_image(
    file: UploadFile = File(None),
    image: UploadFile = File(None),
):
    """Run gesture recognition on an uploaded pet image."""
    target = file if _is_valid_upload(file) else (image if _is_valid_upload(image) else None)
    if not target:
        return JSONResponse({"error": "No image file provided."}, status_code=400)
    return await handle_image_upload(target)


@app.post("/predict_video")
async def predict_video(
    video: UploadFile = File(None),
    file: UploadFile = File(None),
):
    """Run the visual fusion pipeline on an uploaded video."""
    target = video if _is_valid_upload(video) else (file if _is_valid_upload(file) else None)
    if not target:
        return JSONResponse({"error": "No video file provided."}, status_code=400)
    return await handle_video_upload(target)


@app.post("/upload")
async def upload_media(
    file: UploadFile = File(...),
):
    """Unified endpoint to analyze an uploaded image, video, or audio file."""
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    ct = (file.content_type or "").lower()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}

    if ext in image_exts or ct.startswith("image/"):
        return await handle_image_upload(file)
    elif ext in video_exts or ct.startswith("video/"):
        return await handle_video_upload(file)
    elif ext in audio_exts or ct.startswith("audio/"):
        return await handle_audio_upload(file)
    else:
        return JSONResponse(
            {"error": f"Unsupported file format '{ext or ct}'. Please upload an image (JPG, PNG, WEBP), video (MP4, MOV, WEBM), or audio (MP3, WAV, OGG, M4A)."},
            status_code=400,
        )


@app.get("/audio_status")
async def audio_status():
    """
    Diagnostic endpoint: is YAMNet actually loaded on this server?

    Hits classify_vocalization() with a throwaway silent buffer first (this
    triggers the lazy _load() in audio.py if it hasn't run yet, same as a
    real request would), then reports audio.get_status() -- whether the
    model loaded, and if not, the exact underlying error (missing
    tensorflow_hub package vs. a blocked/failed request to tfhub.dev vs.
    something else). This is what distinguishes "the model works but this
    clip was quiet" from "the model never loaded, so nothing is ever
    detected" -- which otherwise look identical from the UI.
    """
    import numpy as np
    try:
        classify_vocalization(np.zeros(16000, dtype=np.float32))
    except Exception:
        pass
    return get_audio_status()


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