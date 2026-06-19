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

def calculate_3d_angles(parent_joint, child_joint):
    """Calculates Pitch (Up/Down) and Yaw (Left/Right) in radians."""
    p_x, p_y, p_z = to_blender_space(parent_joint)
    c_x, c_y, c_z = to_blender_space(child_joint)
    dx = c_x - p_x
    dy = c_y - p_y
    dz = c_z - p_z
    #Yaw: Left/Right swing on the X/Y (Depth) plane
    yaw = math.atan2(dx, dy)
    # Pitch: Up/Down swing using the Z axis and horizontal distance
    horizontal_distance = math.sqrt(dx**2 + dy**2)
    pitch = math.atan2(dz, horizontal_distance)
    return pitch, yaw


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

                p_elbow, y_elbow = calculate_3d_angles(elbow, wrist)
                p_shoulder, y_shoulder = calculate_3d_angles(shoulder,elbow)

                dx = wrist.x - elbow.x
                dy = wrist.y - elbow.y
                flat_angle = math.atan2(dy,dx)
                calibrated_angle = flat_angle - (math.pi / 2)
                data = {
                     #'mixamorig:RightForeArm' : [3*(p_elbow), 0.0, 3*(y_elbow)],
                     #'mixamorig:RightArm' : [0.5*(p_shoulder),0.0,0.5*(y_shoulder)]
                    "mixamorig:RightForeArm" : [calibrated_angle, 0.0, 0.0]
                }
                '''
                lm = results.right_hand_landmarks.landmark[elbow]
                ax = math.degrees(math.acos(lm.x/len_calc(lm.x,lm.y,lm.z)))
                ay = math.degrees(math.acos(lm.y/len_calc(lm.x,lm.y,lm.z)))
                az = math.degrees(math.acos(lm.z/len_calc(lm.x,lm.y,lm.z)))
                print(f"X={ax}, Y={ay}, Z={az}")
                '''

                message = json.dumps(data).encode('utf-8')
                print(f"SENDING: {data}")
                udp_socket.sendto(message, blender_address)

        cv2.imshow('WebCam', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()
