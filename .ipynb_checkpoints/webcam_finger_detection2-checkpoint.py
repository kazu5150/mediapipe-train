# from operator import le
import mediapipe as mp
import cv2
import logging
import numpy as np


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
                # 人差し指と親指の先端のランドマークを取得します
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                # ランドマークの座標を画像の解像度にマッピングします
                image_height, image_width, _ = image.shape
                x_index = int(index_finger_tip.x * image_width)
                y_index = int(index_finger_tip.y * image_height)
                x_thumb = int(thumb_tip.x * image_width)
                y_thumb = int(thumb_tip.y * image_height)
                
                # 人差し指と親指の先端を線で結びます
                cv2.line(image, (x_index, y_index), (x_thumb, y_thumb), (255, 0, 0), 2)
                
                # 人差し指と親指の距離を計算します
                distance = np.sqrt((x_index - x_thumb) ** 2 + (y_index - y_thumb) ** 2)
                
                # 線の中点に距離を描画します
                middle_point = ((x_index + x_thumb) // 2, (y_index + y_thumb) // 2)
                cv2.putText(image, str(int(distance)), middle_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # 親指と人差し指が閾値以内に近づいた場合は「TOUCH!!」を表示
                threshold = 50  # ある閾値
                if distance < threshold:
                    cv2.putText(image, "TOUCH!!", (-200, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

                # 人差し指と親指の先端にランドマークを描画
                cv2.circle(image, (x_index, y_index), 5, (0, 255, 0), -1)
                cv2.circle(image, (x_thumb, y_thumb), 5, (0, 255, 0), -1)

        cv2.imshow("hand detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap_file.release()

logging.debug("プログラム終了")
