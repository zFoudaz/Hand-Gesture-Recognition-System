import streamlit as st
import cv2 
import numpy as np 
import mediapipe as mp
import pickle

st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🖐️",
    layout="wide",
)
st.title("Sign Language Detection")
st.write("Real-time sign language recognition using AI.")


with st.sidebar:
    st.header("⚙️ Controls")

    start_btn = st.button("▶️ Start Detection")
    stop_btn = st.button("⏹️ Stop Detection")



frame_placeholder = st.empty()

with open('model.pkl', 'rb') as f:
    rf = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
stopped=True

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

                # Get wrist to normalize relative positions
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
                d=np.array(data_x+data_y)
                d=d.reshape(1,-1)
                predicted = rf.predict(d)
                # print(predicted)
                cv2.putText(frame, f"{predicted}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)

            # cv2.imshow("Hand Tracking", frame)
            frame_placeholder.image(frame,channels='BGR')

            if cv2.waitKey(1) & 0xFF == ord('q') or stop_btn:
                stopped=True
                break

    cap.release()
    cv2.destroyAllWindows()
