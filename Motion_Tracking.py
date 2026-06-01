import cv2
import numpy as np
import mediapipe as mp
import socket
import json

#def angle_calc()

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
blender_address = ("127.0.0.1", 5052)

vid = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(img,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

        if results.right_hand_landmarks:
            index_tip = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            tip_x = index_tip.x
            tip_y = index_tip.y
            tip_z= index_tip.z
            print(f"Index Finger Tip -> X: {tip_x:.3f} | Y: {tip_y:.3f} | Z: {tip_z:.3f}")

        cv2.imshow('WebCam', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()
