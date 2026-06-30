import cv2
import numpy as np
import mediapipe as mp
import socket
import json
import math


def to_blender_space(landmark):
    """Converts MediaPipe screen coordinates to Blender 3D coordinates."""
    bx = landmark.x
    by = -landmark.z
    bz = landmark.y
    return bx,by,bz

def calculate_2d_angle(parent_joint, child_joint):
    dx = child_joint.x - parent_joint.x
    dy = child_joint.y - parent_joint.y
    
    # Calculate pure 2D angle based only on stable screen X and Y
    angle = math.atan2(dy, dx)
    return angle


def len_calc(x,y,z):
     length = math.sqrt(x**2 + y**2 + z**2)
     return length
     

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
        #mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        #mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

        if results.pose_landmarks:
                '''index_tip = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]'''
                elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
                shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

                #p_elbow, y_elbow = calculate_3d_angles(elbow, wrist)
                r_shoulder = calculate_2d_angle(shoulder,elbow)
                r_elbow = calculate_2d_angle(elbow,wrist)

                calibrated_r_shoulder = r_shoulder - (math.pi)/2
                calibrated_r_elbow = r_elbow - (math.pi)/2
               


                data = {
                     #'mixamorig:RightForeArm' : [3*(p_elbow), 0.0, 3*(y_elbow)],
                    'mixamorig:RightArm' : [calibrated_r_shoulder,0.0,0],
                    "mixamorig:RightForeArm" : [calibrated_r_elbow, 0.0, 0.0]
                }
                

                message = json.dumps(data).encode('utf-8')
                print(f"SENDING: {data}")
                udp_socket.sendto(message, blender_address)

        cv2.imshow('WebCam', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()
