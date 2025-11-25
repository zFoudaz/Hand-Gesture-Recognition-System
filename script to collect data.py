import cv2
import mediapipe as mp
import csv

labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'I love you',
       'OK', 'Telephone', 'Very Good', 'Please', 'Hello']

label = labels[11]
num = 0
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

csv_file = f"{label}.csv"

cap = cv2.VideoCapture(0)

current_label = None

f=open(csv_file, "a", newline="")
writer = csv.writer(f)
header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
writer = csv.writer(f)
writer.writerow(header)

while True:
    _, frame = cap.read()
    h,w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        current_label = label
        print(f"Recording label: {current_label}")
    elif key == ord('q') or num > 350:
        break


    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = hand_landmarks.landmark


        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y

        data_x = []
        data_y = []

        for lm in landmarks:
            norm_x = lm.x - wrist_x
            norm_y = lm.y - wrist_y
            data_x.append(norm_x)
            data_y.append(norm_y)


        if current_label is not None:
            row = [current_label] + data_x + data_y
            writer.writerow(row)
            num+=1

            cv2.putText(frame, f"Saved sample for label {current_label}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"press R when you ready. press Q when you are done",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(frame, f"label: {label}",
                (w-170, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


    cv2.imshow("Dataset Recorder", frame)

cap.release()
cv2.destroyAllWindows()

print("Dataset saved to:", csv_file)
f.close()