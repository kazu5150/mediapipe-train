# from operator import le
import mediapipe as mp
import cv2
import logging


logging.basicConfig(level=logging.INFO,
    format=" %(asctime)s - %(levelname)s - %(message)s")
logging.debug("プログラム開始")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 0, 255))

cap_file = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=4,
    min_detection_confidence=0.5,
    static_image_mode=False) as hands_detection:

    while cap_file.isOpened():
        success, image = cap_file.read()
        if not success:
            print("empty camera frame")
            continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands_detection.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mark_drawing_spec,
                    connection_drawing_spec=mesh_drawing_spec
                    )

        cv2.imshow("hand detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap_file.release()

logging.debug("プログラム終了")
