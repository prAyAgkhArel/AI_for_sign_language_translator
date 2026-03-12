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


st.set_page_config(page_title="ASL Word Recognition", layout="centered")


WORD_MODEL_PATH  = r"C:\Users\praya\Desktop\Custom Dataset\asl_cnn_final_noface.keras"
WORD_LABELS_PATH = r"C:\Users\praya\Desktop\Custom Dataset\class_names_noface.json"

WORD_SEQ_LEN = 30
WORD_FEATURES = 258


MIN_WORD_CONF = 0.75
MIN_MARGIN = 0.25
MIN_MOTION = 0.004
NONE_LABEL = "NONE"

MIN_WORD_CONF = 0.75
MIN_MARGIN = 0.25
MIN_MOTION = 0.004

STABLE_FRAMES = 6
COOLDOWN_FRAMES = 20

START_THRESH = 0.002
END_THRESH   = 0.0008


@st.cache_resource
def load_word_model():
    return tf.keras.models.load_model(WORD_MODEL_PATH, compile=False)


@st.cache_resource
def load_labels(path):
    with open(path) as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    else:
        return [obj[k] for k in sorted(obj.keys(), key=lambda x:int(x))]


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


# -----------------------------
# Hand-focused motion detection
# -----------------------------
def hand_motion(prev_kp, curr_kp):

    left_wrist = 132
    right_wrist = 132 + 63

    left_index = 132 + 8*3
    right_index = 132 + 63 + 8*3

    ids = [
        left_wrist,left_wrist+1,left_wrist+2,
        right_wrist,right_wrist+1,right_wrist+2,
        left_index,left_index+1,left_index+2,
        right_index,right_index+1,right_index+2
    ]

    prev = prev_kp[ids]
    curr = curr_kp[ids]

    return float(np.mean(np.abs(curr-prev)))


class ASLWordProcessor(VideoProcessorBase):

    def __init__(self, model, labels, show_debug=False):

        self.model = model
        self.labels = labels
        self.show_debug = show_debug

        

        self.sign_buffer = []

        self.motion_history = deque(maxlen=5)

        self.prob_buffer = deque(maxlen=STABLE_FRAMES)
        
        self.seq_len = WORD_SEQ_LEN
        self.seq = deque(maxlen=self.seq_len)

        self.state = "IDLE"
        self.cooldown = 0

        self.last_word = None
        self.last_conf = 0

        self.prev_time = time.time()
        self.fps = 0

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        
        self.vote_buffer = deque(maxlen=10)
        self.pred_counter = 0
        
        self.motion_counter = 0
        self.idle_counter = 0


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
                # ======================
        # STATE MACHINE
        # ======================

        if self.state == "IDLE":

            if avg_motion > START_THRESH:
                self.motion_counter += 1
            else:
                self.motion_counter = 0

            if self.motion_counter >= 4:
                self.state = "SIGNING"
                self.sign_buffer = []   # IMPORTANT FIX
                self.vote_buffer.clear()
                self.motion_counter = 0
                self.last_word = None


        elif self.state == "SIGNING":

            self.sign_buffer.append(keypoints)

            # Wait until enough frames collected
            if len(self.sign_buffer) >= self.seq_len:

                seq = self.sign_buffer[-self.seq_len:]
                seq_arr = normalize_sequence(np.array(seq))

                probs = self.model.predict(seq_arr[None,...], verbose=0)[0]
                self.vote_buffer.append(probs)

                # Always compute average if at least 1
                avg_probs = np.mean(self.vote_buffer, axis=0)

                # -------- ALWAYS SHOW TOP 5 ----------
                topk = min(5, len(avg_probs))
                top_idx = np.argsort(avg_probs)[-topk:][::-1]

                if self.show_debug:
                    dbg = " | ".join(
                        f"{self.labels[i]}:{avg_probs[i]:.2f}"
                        for i in top_idx
                    )
                    cv2.putText(img, dbg, (20,80),
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,0),2)
                # -------------------------------------

                # Only apply acceptance when vote buffer stable
                if len(self.vote_buffer) >= 5:

                    idx = int(np.argmax(avg_probs))
                    conf = float(avg_probs[idx])
                    label = self.labels[idx]

                    if avg_probs.size >= 2:
                        top2 = np.partition(avg_probs, -2)[-2:]
                        margin = float(top2.max() - top2.min())
                    else:
                        margin = 0

                    motion_seq = float(np.mean(np.abs(np.diff(seq_arr, axis=0))))

                    if (
                        label != NONE_LABEL and
                        conf >= MIN_WORD_CONF and
                        margin >= MIN_MARGIN and
                        motion_seq >= MIN_MOTION
                    ):
                        self.last_word = label
                        self.last_conf = conf
                        self.state = "COOLDOWN"
                        self.cooldown = COOLDOWN_FRAMES
                        self.sign_buffer = []
                        self.vote_buffer.clear()

            # End signing only if motion low AND enough frames collected
            if avg_motion < END_THRESH and len(self.sign_buffer) < self.seq_len:
                self.state = "IDLE"
                self.sign_buffer = []
                self.vote_buffer.clear()


        elif self.state == "COOLDOWN":

            if self.cooldown > 0:
                self.cooldown -= 1
            else:
                self.state = "IDLE"
                self.last_word = None


        # ======================
        # DISPLAY
        # ======================

        if self.last_word:
            display_text = f"{self.last_word} ({self.last_conf:.2f})"
            color = (0,255,0)
        elif self.state == "SIGNING":
            display_text = "SIGNING..."
            color = (0,0,255)
        else:
            display_text = ""
            color = (0,0,0)

        if display_text:
            cv2.putText(img, display_text, (20,45),
                        cv2.FONT_HERSHEY_SIMPLEX,1,color,2)


        if self.show_debug:

            cv2.putText(img,f"STATE:{self.state}",(20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

            cv2.putText(img,f"MOTION:{avg_motion:.4f}",(20,150),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

            cv2.putText(img,f"BUFFER:{len(self.sign_buffer)}",(20,180),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

            cv2.putText(img,f"FPS:{self.fps:.1f}",(20,210),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)


        return av.VideoFrame.from_ndarray(img, format="bgr24")



st.title("ASL Word Recognition (Segmented)")

model = load_word_model()

labels = load_labels(WORD_LABELS_PATH)

show_debug = st.checkbox("Show debug", value=True)


webrtc_streamer(
    key="asl-word-segmented",
    video_processor_factory=lambda: ASLWordProcessor(model,labels,show_debug),
    media_stream_constraints={"video":True,"audio":False}
)