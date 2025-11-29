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

# ----------------- CUSTOM STYLES -----------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    .status-running {
        background-color: #def7e5;
        color: #1a7f37;
    }
    .status-stopped {
        background-color: #fde0e0;
        color: #b42318;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- HEADER -----------------
st.markdown('<div class="main-title">🖐️ Sign Language Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time sign language recognition using your webcam and AI.</div>', unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "show_webrtc" not in st.session_state:
    st.session_state.show_webrtc = False

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.write("Use these buttons to control the detection session:")

    start_btn = st.button("▶️ Start Detection", use_container_width=True)
    stop_btn = st.button("⏹️ Stop Detection", type="secondary", use_container_width=True)

    if start_btn:
        st.session_state.show_webrtc = True
    if stop_btn:
        st.session_state.show_webrtc = False

    st.markdown("---")
    st.subheader("ℹ️ Tips")
    st.markdown(
        """
        - Make sure your **webcam** is connected  
        - Place your **hand** clearly in front of the camera  
        - Use a **plain background** for better results
        """
    )

# ----------------- MAIN LAYOUT -----------------
left_col, right_col = st.columns([3, 2], vertical_alignment="top")

with left_col:
    st.subheader("📷 Live Preview")
    if not st.session_state.show_webrtc:
        st.info("Click **Start Detection** from the sidebar to begin.")

with right_col:
    st.subheader("🔍 Session Info")
    st.markdown("Current model: `RandomForest (model.pkl)`")
    st.markdown("Max hands: `1`")

    if st.session_state.show_webrtc:
        st.markdown('<span class="status-badge status-running">Status: Running</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-stopped">Status: Stopped</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        The predicted sign will appear **on the video frame** itself.  
        To stop detection, click **Stop Detection** in the sidebar.
        """
    )

# ----------------- ORIGINAL WORKFLOW (MODEL + HAND DETECTOR) -----------------

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

with left_col:
    if st.session_state.show_webrtc:
        webrtc_streamer(
            key="sign-demo",
            video_transformer_factory=HandDetector,
            media_stream_constraints={"video": True, "audio": False},
        )
