import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# -------------------------------
# Load MediaPipe Hand Landmarker
# -------------------------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(options)

# -------------------------------
# OpenCV Setup
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
canvas = None
prev_x, prev_y = None, None

# -------------------------------
# Helper Functions
# -------------------------------
def is_finger_up(lms, tip, pip):
    return lms[tip].y < lms[pip].y

# -------------------------------
# Main Loop
# -------------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Landmark indices
        index_tip = hand[8]
        index_pip = hand[6]

        middle_tip = hand[12]
        middle_pip = hand[10]

        ring_tip = hand[16]
        ring_pip = hand[14]

        pinky_tip = hand[20]
        pinky_pip = hand[18]

        # Gesture detection
        index_up = is_finger_up(hand, 8, 6)
        middle_up = is_finger_up(hand, 12, 10)
        ring_up = is_finger_up(hand, 16, 14)
        pinky_up = is_finger_up(hand, 20, 18)

        # Fist = all fingers down
        fist = not (index_up or middle_up or ring_up or pinky_up)

        ix, iy = int(index_tip.x * w), int(index_tip.y * h)

        if fist:
            canvas[:] = 0
            prev_x, prev_y = None, None
            cv2.putText(frame, "CLEAR", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif index_up and not (middle_up or ring_up or pinky_up):
            if prev_x is None:
                prev_x, prev_y = ix, iy

            cv2.line(canvas, (prev_x, prev_y), (ix, iy),
                     (255, 0, 255), 5)
            prev_x, prev_y = ix, iy

            cv2.circle(frame, (ix, iy), 8, (255, 0, 255), -1)
            cv2.putText(frame, "DRAW", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            prev_x, prev_y = None, None
            cv2.putText(frame, "PAUSE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Merge canvas with frame
    frame = cv2.add(frame, canvas)

    cv2.imshow("AirDraw - MediaPipe Tasks", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
