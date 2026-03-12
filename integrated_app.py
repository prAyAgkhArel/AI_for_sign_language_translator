import os
import json
from collections import deque
import time

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="ASL Translator (Words & Letters)", layout="centered")

# ==========================
# PATHS & CONFIG
# ==========================
# Word Model Paths
WORD_MODEL_PATH  = r"C:\Users\praya\Desktop\Custom Dataset\asl_cnn_final_noface.keras"
WORD_LABELS_PATH = r"C:\Users\praya\Desktop\Custom Dataset\class_names_noface.json"

# Letter Model Paths
LETTER_MODEL_PATH = r"C:\Users\praya\Desktop\VSCODE\AI_for_MAJOR_PROJECT\asl_letters.tflite"
LETTER_LABELS_PATH = r"C:\Users\praya\Desktop\VSCODE\AI_for_MAJOR_PROJECT\letter_labels.txt" 

# Word Model Constants
WORD_SEQ_LEN = 30
WORD_FEATURES = 258
MIN_WORD_CONF = 0.75
MIN_MARGIN = 0.25
MIN_MOTION = 0.004
NONE_LABEL = "NONE"
STABLE_FRAMES = 6
COOLDOWN_FRAMES = 20
START_THRESH = 0.002
END_THRESH   = 0.0008

# ==========================
# RESOURCE LOADERS
# ==========================
@st.cache_resource
def load_word_model():
    return tf.keras.models.load_model(WORD_MODEL_PATH, compile=False)

@st.cache_resource
def load_word_labels(path):
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    return [obj[k] for k in sorted(obj.keys(), key=lambda x:int(x))]

@st.cache_resource
def load_letter_model():
    interpreter = tf.lite.Interpreter(model_path=LETTER_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_letter_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

# ==========================
# HELPER FUNCTIONS
# ==========================
def extract_keypoints_no_face(results):
    if results.pose_landmarks:
        pose = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x,lm.y,lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x,lm.y,lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)

    out = np.concatenate([pose,lh,rh])
    if out.shape[0] != WORD_FEATURES:
        fixed = np.zeros(WORD_FEATURES)
        fixed[:len(out)] = out
        out = fixed
    return out.astype(np.float32)

def normalize_sequence(seq_arr):
    mu = seq_arr.mean()
    sigma = seq_arr.std()
    seq_arr = (seq_arr - mu) / (sigma + 1e-6)
    return np.nan_to_num(seq_arr).astype(np.float32)

def hand_motion(prev_kp, curr_kp):
    prev_hands = prev_kp[132:]
    curr_hands = curr_kp[132:]
    mask = (prev_hands != 0.0) & (curr_hands != 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(curr_hands[mask] - prev_hands[mask])))

# --- Letter Normalization Helpers ---
def normalize_landmarks_letter(lms):
    WRIST, MCP_MIDDLE = 0, 9
    pts = np.array(lms).reshape(21, 3)
    wrist = pts[WRIST].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[MCP_MIDDLE])
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
    if box_area < 0.01 or box_area > 0.8:
        return False
    return True

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

        self.sign_buffer = []
        self.motion_history = deque(maxlen=5)
        self.vote_buffer = deque(maxlen=10)
        self.seq_len = WORD_SEQ_LEN
        self.seq = deque(maxlen=self.seq_len)

        self.state = "IDLE"
        self.cooldown = 0
        self.last_word = None
        self.last_conf = 0
        self.motion_counter = 0
        self.idle_patience = 0

        self.prev_time = time.time()
        self.fps = 0

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def predict_letter(self, results):
        """Extracts hand from holistic results and runs TFLite letter model"""
        # Prioritize right hand, but fallback to left
        hand_lms = None
        if results.right_hand_landmarks:
            hand_lms = results.right_hand_landmarks.landmark
        elif results.left_hand_landmarks:
            hand_lms = results.left_hand_landmarks.landmark

        if not hand_lms:
            return None, 0.0, "No hand detected"

        # Format landmarks for the letter logic
        lms = []
        for lm in hand_lms:
            lms.extend([lm.x, lm.y, lm.z])

        if not is_landmarks_valid_letter(lms):
            return None, 0.0, "Invalid hand shape!"

        input_data = normalize_landmarks_letter(lms)
        
        # Run inference
        input_details = self.letter_interpreter.get_input_details()
        output_details = self.letter_interpreter.get_output_details()
        
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        input_data = np.array(input_data, dtype=input_dtype).reshape(input_shape)

        self.letter_interpreter.set_tensor(input_details[0]['index'], input_data)
        self.letter_interpreter.invoke()
        output_data = self.letter_interpreter.get_tensor(output_details[0]['index'])

        pred_idx = np.argmax(output_data)
        conf = output_data[0][pred_idx]

        if conf < 0.8:
            return None, conf, "Low prediction confidence!"

        return self.letter_labels[pred_idx], conf, ""

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self.fps = 1/dt

        results = self.holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        keypoints = extract_keypoints_no_face(results)
        self.seq.append(keypoints)

        if len(self.seq) > 1:
            motion = hand_motion(self.seq[-2], self.seq[-1])
        else:
            motion = 0

        self.motion_history.append(motion)
        avg_motion = np.mean(self.motion_history)

        # ======================
        # STATE MACHINE (Auto-Switch)
        # ======================
        if self.state == "IDLE":
            if avg_motion > START_THRESH:
                self.motion_counter += 1
            else:
                self.motion_counter = 0

            if self.motion_counter >= 4:
                self.state = "SIGNING"
                self.sign_buffer = []
                self.vote_buffer.clear()
                self.motion_counter = 0
                self.last_word = None

        elif self.state == "SIGNING":
            self.sign_buffer.append(keypoints)

            if len(self.sign_buffer) >= self.seq_len:
                seq = self.sign_buffer[-self.seq_len:]
                seq_arr = normalize_sequence(np.array(seq))

                probs = self.word_model.predict(seq_arr[None,...], verbose=0)[0]
                self.vote_buffer.append(probs)
                avg_probs = np.mean(self.vote_buffer, axis=0)

                # DEBUG for top 5 Words
                topk = min(5, len(avg_probs))
                top_idx = np.argsort(avg_probs)[-topk:][::-1]
                if self.show_debug:
                    dbg = " | ".join(f"{self.word_labels[i]}:{avg_probs[i]:.2f}" for i in top_idx)
                    cv2.putText(img, dbg, (20,80), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,0),2)

                if len(self.vote_buffer) >= 5:
                    idx = int(np.argmax(avg_probs))
                    conf = float(avg_probs[idx])
                    label = self.word_labels[idx]

                    margin = 0
                    if avg_probs.size >= 2:
                        top2 = np.partition(avg_probs, -2)[-2:]
                        margin = float(top2.max() - top2.min())

                    motion_seq = float(np.mean(np.abs(np.diff(seq_arr, axis=0))))

                    if (label != NONE_LABEL and conf >= MIN_WORD_CONF and 
                        margin >= MIN_MARGIN and motion_seq >= MIN_MOTION):
                        self.last_word = label
                        self.last_conf = conf
                        self.state = "COOLDOWN"
                        self.cooldown = COOLDOWN_FRAMES
                        self.sign_buffer = []
                        self.vote_buffer.clear()

            # End signing if motion stops
            if avg_motion < END_THRESH and len(self.sign_buffer) < self.seq_len:
                self.idle_patience += 1
                if self.idle_patience >= 5:
                    self.state = "IDLE"
                    self.sign_buffer = []
                    self.vote_buffer.clear()
                    self.idle_patience = 0
            else:
                self.idle_patience = 0

        elif self.state == "COOLDOWN":
            if self.cooldown > 0:
                self.cooldown -= 1
            else:
                self.state = "IDLE"


        # ======================
        # DISPLAY LOGIC
        # ======================
        if self.state == "SIGNING":
            display_text = "SIGNING WORD..."
            color = (0, 0, 255) # Red for active recording
        elif self.state == "COOLDOWN" and self.last_word:
            display_text = f"Word: {self.last_word} ({self.last_conf:.2f})"
            color = (0, 255, 0) # Green for successful word
        else:
            # We are IDLE, attempt Letter Prediction
            letter, conf, msg = self.predict_letter(results)
            if letter:
                display_text = f"Letter: {letter} ({conf:.2f})"
                color = (255, 165, 0) # Orange for Letter
            elif msg:
                display_text = msg
                color = (150, 150, 150) # Gray for missing hand / warnings
            else:
                display_text = ""
                color = (0,0,0)

        if display_text:
            cv2.putText(img, display_text, (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if self.show_debug:
            cv2.putText(img, f"STATE:{self.state}", (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            cv2.putText(img, f"MOTION:{avg_motion:.4f}", (20,150), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            cv2.putText(img, f"FPS:{self.fps:.1f}", (20,180), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==========================
# STREAMLIT UI
# ==========================
st.title("ASL Translator (Words & Letters)")
st.write("Hold your hand still for Letter prediction. Move your hands to trigger Word prediction.")

word_model = load_word_model()
word_labels = load_word_labels(WORD_LABELS_PATH)

letter_model = load_letter_model()
letter_labels = load_letter_labels(LETTER_LABELS_PATH)

show_debug = st.checkbox("Show debug info", value=True)

webrtc_streamer(
    key="asl-combined",
    video_processor_factory=lambda: ASLCombinedProcessor(
        word_model, word_labels, 
        letter_model, letter_labels, 
        show_debug
    ),
    media_stream_constraints={"video": True, "audio": False}
)