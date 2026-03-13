
import json
from collections import deque
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

st.set_page_config(page_title="ASL Translator (Letters + Words)", layout="centered")

# ==========================
# PATHS
# ==========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CLASS_DIR = BASE_DIR / "class"

WORD_MODEL_PATH = MODELS_DIR / "asl_word_bilstm_normvel.keras"
WORD_LABELS_PATH = CLASS_DIR / "class_names_noface.json"

LETTER_MODEL_PATH = MODELS_DIR / "asl_letters.tflite"
LETTER_LABELS_PATH = CLASS_DIR / "letter_labels.txt"

# ==========================
# CONFIG
# ==========================
RAW_FEATURES = 258
WORD_SEQ_LEN = 30
WORD_CAPTURE_FRAMES = 36
WORD_TRIM_START = 3
WORD_EXPECTED_FEATURES = RAW_FEATURES * 2  # normalized positions + velocity

MIN_WORD_CONF = 0.70
MIN_MARGIN = 0.12

POSE_SIZE = 33 * 4
HAND_SIZE = 21 * 3


# ==========================
# RESOURCE LOADERS
# ==========================
@st.cache_resource
def load_word_model():
    if not WORD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Word model not found at {WORD_MODEL_PATH}. "
            "Train the updated notebook first, or change the path."
        )
    return tf.keras.models.load_model(WORD_MODEL_PATH, compile=False)


@st.cache_resource
def load_word_labels(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Word labels not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    return [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x))]


@st.cache_resource
def load_letter_model():
    if not LETTER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Letter model not found at {LETTER_MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=str(LETTER_MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_letter_labels(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Letter labels not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ==========================
# HELPERS
# ==========================
def extract_keypoints_no_face(results):
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        pose = np.zeros(POSE_SIZE, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        lh = np.zeros(HAND_SIZE, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        rh = np.zeros(HAND_SIZE, dtype=np.float32)

    out = np.concatenate([pose, lh, rh], axis=0)
    if out.shape[0] != RAW_FEATURES:
        fixed = np.zeros(RAW_FEATURES, dtype=np.float32)
        fixed[: min(RAW_FEATURES, out.shape[0])] = out[: min(RAW_FEATURES, out.shape[0])]
        out = fixed
    return out.astype(np.float32)


def normalize_hand(hand_flat):
    hand = hand_flat.reshape(21, 3).astype(np.float32)
    if np.allclose(hand, 0.0):
        return np.zeros_like(hand_flat, dtype=np.float32)

    wrist = hand[0].copy()
    hand = hand - wrist

    scale = np.linalg.norm(hand[9])  # middle MCP
    if scale < 1e-6:
        valid = np.linalg.norm(hand, axis=1)
        scale = float(np.mean(valid[valid > 0])) if np.any(valid > 0) else 1.0

    hand = hand / (scale + 1e-6)
    return hand.flatten().astype(np.float32)


def normalize_pose(pose_flat):
    pose = pose_flat.reshape(33, 4).astype(np.float32)
    xyz = pose[:, :3].copy()
    vis = pose[:, 3:4].copy()

    if np.allclose(xyz, 0.0):
        return pose_flat.astype(np.float32)

    left_shoulder = xyz[11]
    right_shoulder = xyz[12]

    if np.linalg.norm(left_shoulder) > 0 and np.linalg.norm(right_shoulder) > 0:
        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)
    else:
        center = xyz[0]
        norms = np.linalg.norm(xyz, axis=1)
        positive = norms[norms > 0]
        scale = float(np.mean(positive)) if positive.size else 1.0

    xyz = (xyz - center) / (scale + 1e-6)
    out = np.concatenate([xyz, vis], axis=1).flatten()
    return out.astype(np.float32)


def normalize_frame(raw_frame):
    raw_frame = raw_frame.astype(np.float32).reshape(-1)
    pose = normalize_pose(raw_frame[:POSE_SIZE])
    lh = normalize_hand(raw_frame[POSE_SIZE:POSE_SIZE + HAND_SIZE])
    rh = normalize_hand(raw_frame[POSE_SIZE + HAND_SIZE:])
    return np.concatenate([pose, lh, rh], axis=0).astype(np.float32)


def resample_sequence(seq, target_len=WORD_SEQ_LEN):
    """Resample a variable-length sequence to a fixed length."""
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == target_len:
        return seq
    idx = np.linspace(0, len(seq) - 1, target_len).round().astype(int)
    return seq[idx]


def preprocess_word_sequence(raw_seq):
    """
    raw_seq: (T, 258)
    returns: (30, 516) = normalized positions + velocity
    """
    raw_seq = np.asarray(raw_seq, dtype=np.float32)

    if raw_seq.ndim != 2 or raw_seq.shape[1] != RAW_FEATURES:
        raise ValueError(f"Expected raw sequence shape (T, {RAW_FEATURES}), got {raw_seq.shape}")

    if len(raw_seq) >= WORD_TRIM_START + WORD_SEQ_LEN:
        raw_seq = raw_seq[WORD_TRIM_START:]
    raw_seq = resample_sequence(raw_seq, WORD_SEQ_LEN)

    norm_seq = np.stack([normalize_frame(f) for f in raw_seq], axis=0)
    vel_seq = np.diff(norm_seq, axis=0, prepend=norm_seq[:1])
    feat_seq = np.concatenate([norm_seq, vel_seq], axis=1)
    return np.nan_to_num(feat_seq).astype(np.float32)


# --- Letter model helpers ---
def normalize_landmarks_letter(lms):
    pts = np.array(lms, dtype=np.float32).reshape(21, 3)
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9])  # middle MCP
    if scale < 1e-6:
        scale = np.linalg.norm(pts).mean() + 1e-6
    pts /= scale
    return pts.flatten().astype(np.float32)


def is_landmarks_valid_letter(lms):
    xs = [x for i, x in enumerate(lms) if i % 3 == 0]
    ys = [y for i, y in enumerate(lms) if i % 3 == 1]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    box_area = (max_x - min_x) * (max_y - min_y)
    return 0.01 <= box_area <= 0.8


# ==========================
# VIDEO PROCESSOR
# ==========================
class ASLCombinedProcessor(VideoProcessorBase):
    def __init__(self, word_model, word_labels, letter_interpreter, letter_labels, show_debug=False):
        self.word_model = word_model
        self.word_labels = word_labels
        self.letter_interpreter = letter_interpreter
        self.letter_labels = letter_labels
        self.show_debug = show_debug

        self.mode = "LETTER"
        self.capture_requested = False
        self.capture_active = False
        self.capture_buffer = []

        self.last_word = None
        self.last_word_conf = 0.0
        self.last_top3 = []
        self.last_letter = None
        self.last_letter_conf = 0.0

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def request_word_capture(self):
        self.capture_requested = True

    def clear_last_word(self):
        self.last_word = None
        self.last_word_conf = 0.0
        self.last_top3 = []

    def predict_letter(self, results):
        hand_lms = None
        if results.right_hand_landmarks:
            hand_lms = results.right_hand_landmarks.landmark
        elif results.left_hand_landmarks:
            hand_lms = results.left_hand_landmarks.landmark

        if not hand_lms:
            return None, 0.0, "No hand detected"

        lms = []
        for lm in hand_lms:
            lms.extend([lm.x, lm.y, lm.z])

        if not is_landmarks_valid_letter(lms):
            return None, 0.0, "Invalid hand shape"

        input_data = normalize_landmarks_letter(lms)

        input_details = self.letter_interpreter.get_input_details()
        output_details = self.letter_interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        input_data = np.array(input_data, dtype=input_dtype).reshape(input_shape)

        self.letter_interpreter.set_tensor(input_details[0]["index"], input_data)
        self.letter_interpreter.invoke()
        output_data = self.letter_interpreter.get_tensor(output_details[0]["index"])

        pred_idx = int(np.argmax(output_data))
        conf = float(output_data[0][pred_idx])

        if conf < 0.80:
            return None, conf, "Low letter confidence"

        return self.letter_labels[pred_idx], conf, ""

    def predict_word(self):
        seq = preprocess_word_sequence(np.array(self.capture_buffer, dtype=np.float32))
        probs = self.word_model.predict(seq[None, ...], verbose=0)[0]

        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [(self.word_labels[i], float(probs[i])) for i in top_idx]

        conf1 = float(probs[top_idx[0]])
        conf2 = float(probs[top_idx[1]]) if len(top_idx) > 1 else 0.0
        margin = conf1 - conf2

        self.last_top3 = top3

        if conf1 < MIN_WORD_CONF or margin < MIN_MARGIN:
            self.last_word = "UNKNOWN"
            self.last_word_conf = conf1
        else:
            self.last_word = self.word_labels[top_idx[0]]
            self.last_word_conf = conf1

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        if self.mode == "WORD":
            keypoints = extract_keypoints_no_face(results)

            if self.capture_requested and not self.capture_active:
                self.capture_active = True
                self.capture_requested = False
                self.capture_buffer = []
                self.last_top3 = []

            if self.capture_active:
                self.capture_buffer.append(keypoints)
                remaining = WORD_CAPTURE_FRAMES - len(self.capture_buffer)
                cv2.putText(
                    img,
                    f"Recording word clip... {max(remaining, 0)}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                if len(self.capture_buffer) >= WORD_CAPTURE_FRAMES:
                    self.capture_active = False
                    self.predict_word()

            if not self.capture_active:
                if self.last_word:
                    color = (0, 255, 0) if self.last_word != "UNKNOWN" else (0, 165, 255)
                    cv2.putText(
                        img,
                        f"Word: {self.last_word} ({self.last_word_conf:.2f})",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )
                else:
                    cv2.putText(
                        img,
                        "Word mode: press 'Start word capture'",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (200, 200, 200),
                        2,
                    )

            if self.show_debug and self.last_top3:
                y = 80
                for label, conf in self.last_top3:
                    cv2.putText(
                        img,
                        f"{label}: {conf:.2f}",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 0),
                        2,
                    )
                    y += 28

        else:
            letter, conf, msg = self.predict_letter(results)
            if letter:
                self.last_letter = letter
                self.last_letter_conf = conf
                cv2.putText(
                    img,
                    f"Letter: {letter} ({conf:.2f})",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 165, 0),
                    2,
                )
            elif msg:
                cv2.putText(
                    img,
                    msg,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (150, 150, 150),
                    2,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==========================
# STREAMLIT UI
# ==========================
st.title("ASL Translator (Letters + Words)")
st.write(
    "Use Letter mode for single-hand alphabet prediction. "
    "Use Word mode and press the button to record one short word clip."
)

word_model = load_word_model()
word_labels = load_word_labels(WORD_LABELS_PATH)
letter_model = load_letter_model()
letter_labels = load_letter_labels(LETTER_LABELS_PATH)

left_col, right_col = st.columns(2)
with left_col:
    mode = st.radio("Mode", ["LETTER", "WORD"], horizontal=True)
with right_col:
    show_debug = st.checkbox("Show debug info", value=True)

start_capture = False
clear_word = False
if mode == "WORD":
    c1, c2 = st.columns(2)
    with c1:
        start_capture = st.button("Start word capture", type="primary", use_container_width=True)
    with c2:
        clear_word = st.button("Clear last word", use_container_width=True)

webrtc_ctx = webrtc_streamer(
    key="asl-combined-updated",
    video_processor_factory=lambda: ASLCombinedProcessor(
        word_model=word_model,
        word_labels=word_labels,
        letter_interpreter=letter_model,
        letter_labels=letter_labels,
        show_debug=show_debug,
    ),
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.mode = mode
    webrtc_ctx.video_processor.show_debug = show_debug
    if start_capture:
        webrtc_ctx.video_processor.request_word_capture()
    if clear_word:
        webrtc_ctx.video_processor.clear_last_word()

st.caption(
    "Updated behavior: no fake NONE class, manual word capture, trimmed word clip, "
    "top-3 debug output, and word preprocessing matched to the updated training notebook."
)
