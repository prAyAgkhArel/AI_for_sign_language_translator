import os
import json
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================
# STREAMLIT CONFIG (must be the first Streamlit call)
# =========================
st.set_page_config(page_title="ASL Word Recognition", layout="centered")


# =========================
# PATHS (update if needed)
# =========================
WORD_MODEL_PATH  = r"C:\Users\praya\Desktop\Custom Dataset\asl_cnn_final_noface.keras"
WORD_LABELS_PATH = r"C:\Users\praya\Desktop\Custom Dataset\class_names_noface.json"


# =========================
# SETTINGS
# =========================
WORD_SEQ_LEN   = 30
WORD_FEATURES  = 258     # NO-FACE FEATURES: pose(33*4) + lh(21*3) + rh(21*3) = 132+63+63 = 258
NONE_LABEL     = "NONE"  # must match your training label exactly



# Extra stability gates (tune these)
MARGIN_THRESH = 0.20     # top1 - top2 probability
MOTION_THRESH = 0.008    # mean abs diff across time (after normalization)

WORD_SEQ_LEN  = 30
WORD_FEATURES = 258
NONE_LABEL    = "NONE"

# 1) Minimum confidence for accepting a word
MIN_WORD_CONF = 0.70  # try 0.45–0.70

# 2) Stability: how many consecutive predictions to average
STABLE_FRAMES = 6     # try 3–6 (higher = steadier, slower)

# 3) Cooldown: frames to wait after detecting a word (prevents repeats)
COOLDOWN_FRAMES = 20  # try 10–25

# 4) Margin gate: top1 - top2 must be large enough
MIN_MARGIN = 0.20      # try 0.15–0.30

# 5) Motion gate: reject if user is not moving/signing
MIN_MOTION = 0.01     # try 0.005–0.02
# =========================
# LOADERS
# =========================
@st.cache_resource
def load_word_model():
    # compile=False is fine for inference and loads faster
    return tf.keras.models.load_model(WORD_MODEL_PATH, compile=False)


@st.cache_resource
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Allow either list ["HELLO", ...] or dict {"0":"HELLO", ...}
    if isinstance(obj, list):
        labels = obj
    elif isinstance(obj, dict):
        keys = sorted(obj.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        labels = [obj[k] for k in keys]
    else:
        raise TypeError(f"Labels JSON must be list or dict, got {type(obj)}")

    labels = [str(x) for x in labels]
    return labels


# =========================
# KEYPOINT EXTRACTION (NO FACE)
# =========================
def extract_keypoints_no_face(results) -> np.ndarray:
    # Pose: 33*4 = 132
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # Left hand: 21*3 = 63
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # Right hand: 21*3 = 63
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    out = np.concatenate([pose, lh, rh], axis=0)

    # Hard-enforce length
    if out.shape[0] != WORD_FEATURES:
        fixed = np.zeros(WORD_FEATURES, dtype=np.float32)
        fixed[:min(WORD_FEATURES, out.shape[0])] = out[:min(WORD_FEATURES, out.shape[0])]
        out = fixed

    return out


def normalize_sequence(seq_arr: np.ndarray) -> np.ndarray:
    """
    Match training normalization:
      X = (X - mean(axis=(1,2))) / (std(axis=(1,2)) + 1e-6)
    Here seq_arr is (30, 258).
    """
    mu = float(seq_arr.mean())
    sigma = float(seq_arr.std())
    seq_arr = (seq_arr - mu) / (sigma + 1e-6)
    seq_arr = np.nan_to_num(seq_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return seq_arr.astype(np.float32, copy=False)


# =========================
# VIDEO PROCESSOR (WORD ONLY)
# =========================
class ASLWordProcessor(VideoProcessorBase):
    def __init__(self, model, labels, show_debug=False):
        self.model = model
        self.labels = labels
        self.show_debug = show_debug

        self.seq = deque(maxlen=WORD_SEQ_LEN)
        self.prob_buffer = deque(maxlen=STABLE_FRAMES)
        self.cooldown = 0

        self.mp_holistic = mp.solutions.holistic
        # Holistic still runs face internally, but we ignore it in features.
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,
            refine_face_landmarks=False,
        )

        self.last_word = None
        self.last_conf = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1) Extract keypoints
        results = self.holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        keypoints = extract_keypoints_no_face(results)
        self.seq.append(keypoints)

        # 2) Cooldown decrement (once)
        if self.cooldown > 0:
            self.cooldown -= 1
            if self.cooldown == 0:
                self.last_word = None

        # 3) Overlay text baseline
        if self.last_word is not None:
            display_text = f"WORD: {self.last_word} ({self.last_conf:.2f})"
            display_color = (0, 255, 0)
        else:
            display_text = "SIGNING..."
            display_color = (0, 0, 255)

        # 4) Predict when ready
        if len(self.seq) == WORD_SEQ_LEN:
            seq_arr = np.array(self.seq, dtype=np.float32)  # (30, 258)
            seq_arr = normalize_sequence(seq_arr)

            X = seq_arr[None, ...]  # (1, 30, 258)
            probs = self.model.predict(X, verbose=0)[0]      # (num_classes,)
            self.prob_buffer.append(probs)

            if len(self.prob_buffer) == STABLE_FRAMES and self.cooldown == 0:
                avg_probs = np.mean(self.prob_buffer, axis=0)

                idx = int(np.argmax(avg_probs))
                conf = float(avg_probs[idx])
                label = self.labels[idx] if idx < len(self.labels) else str(idx)

                # margin: top1 - top2
                if avg_probs.size >= 2:
                    top2 = np.partition(avg_probs, -2)[-2:]
                    margin = float(top2.max() - top2.min())
                else:
                    margin = 0.0

                # motion: mean abs diff over time
                motion = float(np.mean(np.abs(np.diff(seq_arr, axis=0))))

                # NONE = reject always
                if label != NONE_LABEL:
                    if conf >= MIN_WORD_CONF and margin >= MIN_MARGIN and motion >= MIN_MOTION:

                        self.last_word = label
                        self.last_conf = conf
                        self.cooldown = COOLDOWN_FRAMES
                        self.seq.clear()
                        self.prob_buffer.clear()

                # Debug overlay
                if self.show_debug:
                    topk = 5 if avg_probs.size >= 5 else avg_probs.size
                    top_idx = np.argsort(avg_probs)[-topk:][::-1]
                    dbg = " | ".join(
                        f"{self.labels[i]}:{avg_probs[i]:.2f}" if i < len(self.labels) else f"{i}:{avg_probs[i]:.2f}"
                        for i in top_idx
                    )
                    cv2.putText(img, dbg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.putText(img, display_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, display_color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# STREAMLIT UI
# =========================
st.title("ASL Word Recognition (Word-Only, No-Face)")

st.write("Model path exists:", os.path.exists(WORD_MODEL_PATH))
st.write("Labels path exists:", os.path.exists(WORD_LABELS_PATH))

if not os.path.exists(WORD_MODEL_PATH) or not os.path.exists(WORD_LABELS_PATH):
    st.error("Model or labels not found. Update WORD_MODEL_PATH / WORD_LABELS_PATH.")
    st.stop()

model = load_word_model()
labels = load_labels(WORD_LABELS_PATH)

st.write("Num labels:", len(labels))
st.write("Model output classes:", model.output_shape[-1])

if model.output_shape[-1] != len(labels):
    st.warning(
        "⚠️ Model output classes != number of labels.\n"
        "Fix your labels JSON by saving it from the same 'usable_words' used in training."
    )

show_debug = st.checkbox("Show debug top predictions", value=False)

st.caption("Predicts one word when stable + confident. Uses NONE to reject 'no sign'.")

st.subheader("Prediction Settings")

MIN_WORD_CONF = st.slider("Min confidence", 0.0, 1.0, float(MIN_WORD_CONF), 0.01)
MIN_MARGIN    = st.slider("Min margin (top1-top2)", 0.0, 1.0, float(MIN_MARGIN), 0.01)
MIN_MOTION    = st.slider("Min motion", 0.0, 0.05, float(MIN_MOTION), 0.001)

STABLE_FRAMES = st.slider("Stable frames", 1, 10, int(STABLE_FRAMES), 1)
COOLDOWN_FRAMES = st.slider("Cooldown frames", 0, 60, int(COOLDOWN_FRAMES), 1)


webrtc_streamer(
    key="asl-word-only-noface",
    video_processor_factory=lambda: ASLWordProcessor(model, labels, show_debug=show_debug),
    media_stream_constraints={"video": True, "audio": False},
)
