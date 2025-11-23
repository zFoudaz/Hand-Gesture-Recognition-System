import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🖐️",
    layout="wide",
)

st.title("Sign Language Detection")
st.write("Real-time sign language recognition using AI.")

with open("model.pkl", "rb") as f:
    rf = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class HandDetector(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = hand.landmark

            wrist_x = landmarks[0].x
            wrist_y = landmarks[0].y

            data_x, data_y = [], []

            for lm in landmarks:
                data_x.append(lm.x - wrist_x)
                data_y.append(lm.y - wrist_y)

            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            d = np.array(data_x + data_y).reshape(1, -1)
            predicted = rf.predict(d)[0]

            cv2.putText(
                img,
                str(predicted),
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
            )

        return img



webrtc_streamer(
    key="sign-demo",
    video_transformer_factory=HandDetector,
    media_stream_constraints={"video": True, "audio": False},
)
