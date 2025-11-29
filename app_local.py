import streamlit as st
import cv2 
import numpy as np 
import mediapipe as mp
import pickle

# ----------------- PAGE CONFIG -----------------
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

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.write("Use these buttons to control the detection session:")

    start_btn = st.button("▶️ Start Detection", use_container_width=True)
    stop_btn = st.button("⏹️ Stop Detection", type="secondary", use_container_width=True)

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
    frame_placeholder = st.empty()

with right_col:
    st.subheader("🔍 Session Info")
    # NOTE: purely visual, not affecting your workflow
    st.markdown("Current model: `RandomForest (model.pkl)`")
    st.markdown("Max hands: `1`")

    # simple status indicator (visual only)
    # (real state is still controlled by your variables below)
    stopped = True
    if start_btn:
        stopped = False
    if stop_btn:
        stopped = True

    if stopped:
        st.markdown('<span class="status-badge status-stopped">Status: Stopped</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-running">Status: Running</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        The predicted sign will appear **on the video frame** itself.  
        To stop detection, click **Stop Detection** or press `q` in the window.
        """
    )



with open('model.pkl', 'rb') as f:
    rf = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
stopped = True 

if start_btn and stopped:
    stopped = False
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            _, frame = cap.read()

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = hand.landmark

                wrist_x = landmarks[0].x
                wrist_y = landmarks[0].y

                data_x = []
                data_y = []

                for lm in landmarks:
                    norm_x = lm.x - wrist_x
                    norm_y = lm.y - wrist_y
                    data_x.append(norm_x)
                    data_y.append(norm_y)

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                d = np.array(data_x + data_y)
                d = d.reshape(1, -1)
                predicted = rf.predict(d)

                cv2.putText(
                    frame,
                    f"{predicted}",
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )

            frame_placeholder.image(frame, channels='BGR')

            if cv2.waitKey(1) & 0xFF == ord('q') or stop_btn:
                stopped = True
                break

    cap.release()
    cv2.destroyAllWindows()
