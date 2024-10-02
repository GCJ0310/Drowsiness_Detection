import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import os

# Constants
EYE_AR_THRESH = 0.25
CLOSED_FRAMES = 20  # Number of consecutive frames indicating drowsiness

# Initialize frame counters and blink detector
blink_count = 0

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Function to get eye landmarks from face mesh
def get_eye_landmarks(landmarks, eye_indices):
    return [(landmarks[i].x, landmarks[i].y) for i in eye_indices]

# Function to enlarge the eye region for better visibility
def enlarge_eye(eye, scale=1.5):
    eye_center = np.mean(eye, axis=0)  # Calculate the center of the eye
    return (eye - eye_center) * scale + eye_center  # Scale eye points and move them back to center

# Indices for left and right eyes from the 468 face landmarks in MediaPipe
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Start video capture
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame for face mesh
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get left and right eye landmarks
            left_eye_landmarks = get_eye_landmarks(face_landmarks.landmark, LEFT_EYE_IDX)
            right_eye_landmarks = get_eye_landmarks(face_landmarks.landmark, RIGHT_EYE_IDX)
            
            # Convert landmarks from normalized coordinates to image coordinates
            h, w, _ = frame.shape
            left_eye = np.array([[int(x * w), int(y * h)] for (x, y) in left_eye_landmarks], dtype='int')
            right_eye = np.array([[int(x * w), int(y * h)] for (x, y) in right_eye_landmarks], dtype='int')

            # Enlarge the eyes for better visibility
            left_eye = enlarge_eye(left_eye, scale=1.8)  # Adjust scale value for desired size
            right_eye = enlarge_eye(right_eye, scale=1.8)

            # Compute the EAR for both eyes
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)

            left_eye = np.array(left_eye, dtype=np.int32)
            right_eye = np.array(right_eye, dtype=np.int32)
            
            # Average EAR for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Check if EAR is below the threshold, indicating a blink or drowsiness
            if ear < EYE_AR_THRESH:
                blink_count += 1

                # If blink count exceeds frames threshold, send drowsiness alert
                if blink_count >= CLOSED_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                blink_count = 0

            # Draw the computed EAR on the frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw enlarged eye contours
            cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
