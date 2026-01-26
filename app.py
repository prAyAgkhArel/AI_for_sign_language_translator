import os
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================
# PATHS (your paths)
# =========================
WORD_MODEL_PATH = r"C:\Users\praya\Desktop\VSCODE\AI_for_MAJOR_PROJECT\models\asl_lstm_best.keras"
WORD_LABELS_PATH = r"labels.txt"

LETTER_MODEL_PATH = r"asl_letters.tflite"
LETTER_LABELS_PATH = r"letter_labels.txt"


# =========================
# SETTINGS
# =========================
WORD_SEQ_LEN = 30
WORD_FEATURES = 1662

WORD_CONF = 0.70
STABLE_FRAMES = 6
COOLDOWN_FRAMES = 12

LETTER_HAND_SCORE_MIN = 0.50
LETTER_CONF_MIN = 0.80


# =========================
# LOADERS (MAIN THREAD)
# =========================
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


@st.cache_resource
def load_word_model_and_labels(model_path: str, labels_path: str):
    model = tf.keras.models.load_model(model_path, compile=False)
    labels = load_labels(labels_path)
    return model, labels


@st.cache_resource
def load_letter_tflite_and_labels(tflite_path: str, labels_path: str):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    labels = load_labels(labels_path)
    return interpreter, in_details, out_details, labels


# =========================
# KEYPOINTS (WORD MODEL)
# =========================
def extract_keypoints_holistic(results) -> np.ndarray:
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                        for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.face_landmarks.landmark], dtype=np.float32).flatten()
    else:
        face = np.zeros(468 * 3, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                      for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                      for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    # Your training pipeline expects 1662 total.
    # NOTE: your training used (face + pose + lh + rh) = 1404 + 132 + 63 + 63 = 1662
    return np.concatenate([face, pose, lh, rh], axis=0).astype(np.float32)


# =========================
# LETTER HELPERS (TFLITE)
# =========================
def normalize_landmarks_63(lms_63):
    WRIST, MCP_MIDDLE = 0, 9
    pts = np.array(lms_63, dtype=np.float32).reshape(21, 3)
    wrist = pts[WRIST].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[MCP_MIDDLE])
    if scale < 1e-6:
        scale = np.linalg.norm(pts).mean() + 1e-6
    pts /= scale
    return pts.flatten().astype(np.float32)


def is_landmarks_valid_63(lms_63):
    xs = [x for i, x in enumerate(lms_63) if i % 3 == 0]
    ys = [y for i, y in enumerate(lms_63) if i % 3 == 1]
    box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return (box_area >= 0.01) and (box_area <= 0.8)


# =========================
# VIDEO PROCESSOR
# =========================
class ASLProcessor(VideoProcessorBase):
    def __init__(self, word_model, word_labels, letter_interpreter, letter_in, letter_out, letter_labels):
        self.word_model = word_model
        self.word_labels = word_labels

        self.letter_interpreter = letter_interpreter
        self.letter_in = letter_in
        self.letter_out = letter_out
        self.letter_labels = letter_labels

        # MediaPipe objects inside processor are OK
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)

        self.seq_word = deque(maxlen=WORD_SEQ_LEN)
        self.word_votes = deque(maxlen=STABLE_FRAMES)
        self.cooldown = 0

    def _predict_word(self):
        Xw = np.expand_dims(np.array(self.seq_word, dtype=np.float32), axis=0)  # (1,30,1662)
        probs = self.word_model.predict(Xw, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.word_labels[idx] if idx < len(self.word_labels) else str(idx)
        return label, conf

    def _predict_letter(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None, 0.0

        handedness = results.multi_handedness[0].classification[0]
        score = float(handedness.score)
        if score < LETTER_HAND_SCORE_MIN:
            return None, 0.0

        lms_obj = results.multi_hand_landmarks[0]
        lms = []
        for lm in lms_obj.landmark:
            lms.extend([lm.x, lm.y, lm.z])

        if not is_landmarks_valid_63(lms):
            return None, 0.0

        x = normalize_landmarks_63(lms)

        in_idx = self.letter_in[0]["index"]
        out_idx = self.letter_out[0]["index"]

        in_shape = self.letter_in[0]["shape"]   # usually (1,63)
        in_dtype = self.letter_in[0]["dtype"]
        x = np.array(x, dtype=in_dtype).reshape(in_shape)

        self.letter_interpreter.set_tensor(in_idx, x)
        self.letter_interpreter.invoke()
        out = self.letter_interpreter.get_tensor(out_idx)[0]  # (num_letters,)

        idx = int(np.argmax(out))
        conf = float(out[idx])
        label = self.letter_labels[idx] if idx < len(self.letter_labels) else str(idx)
        return label, conf

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Word keypoints from holistic
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        holo_res = self.holistic.process(img_rgb)
        keypoints = extract_keypoints_holistic(holo_res)

        # enforce length 1662
        if keypoints.shape[0] != WORD_FEATURES:
            if keypoints.shape[0] < WORD_FEATURES:
                keypoints = np.pad(keypoints, (0, WORD_FEATURES - keypoints.shape[0]))
            else:
                keypoints = keypoints[:WORD_FEATURES]

        self.seq_word.append(keypoints)

        if self.cooldown > 0:
            self.cooldown -= 1

        # Default output
        display_text = "LETTER: (no confident letter)"
        display_color = (0, 0, 255)

        # Try accept WORD
        word_accepted = False
        if len(self.seq_word) == WORD_SEQ_LEN:
            w_label, w_conf = self._predict_word()
            self.word_votes.append(w_label)

            stable = (len(self.word_votes) == STABLE_FRAMES) and (len(set(self.word_votes)) == 1)

            if self.cooldown == 0 and stable and w_conf >= WORD_CONF:
                display_text = f"WORD: {w_label} ({w_conf:.2f})"
                display_color = (0, 255, 0)
                self.cooldown = COOLDOWN_FRAMES
                self.word_votes.clear()
                word_accepted = True

        # Fallback to letter
        if not word_accepted:
            l_label, l_conf = self._predict_letter(img)
            if l_label is not None and l_conf >= LETTER_CONF_MIN:
                display_text = f"LETTER: {l_label} ({l_conf:.2f})"
                display_color = (0, 255, 0)

        cv2.putText(img, display_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, display_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="ASL Auto Word/Letter", layout="centered")
st.title("ASL Translator (Auto Word / Letter)")

# basic file checks (helps you catch wrong paths quickly)
for p in [WORD_MODEL_PATH, WORD_LABELS_PATH, LETTER_MODEL_PATH, LETTER_LABELS_PATH]:
    if not os.path.exists(p):
        st.error(f"Missing file: {p}")
        st.stop()

word_model, word_labels = load_word_model_and_labels(WORD_MODEL_PATH, WORD_LABELS_PATH)
letter_interpreter, letter_in, letter_out, letter_labels = load_letter_tflite_and_labels(LETTER_MODEL_PATH, LETTER_LABELS_PATH)

st.caption("WORD is shown only when stable + confident; otherwise LETTER is shown.")

webrtc_streamer(
    key="asl-auto",
    video_processor_factory=lambda: ASLProcessor(
        word_model, word_labels,
        letter_interpreter, letter_in, letter_out, letter_labels
    ),
    media_stream_constraints={"video": True, "audio": False},
)
