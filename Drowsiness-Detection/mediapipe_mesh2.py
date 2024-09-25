import cv2
import mediapipe as mp
import math

def euclidean_distance(point1, point2):

    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def eye_aspect_ratio(eye_landmarks):

    # Calculate the Euclidean distances between the sets of vertical eye landmarks
    vertical_dist1 = euclidean_distance(eye_landmarks[1], eye_landmarks[9])  # 160-144
    vertical_dist2 = euclidean_distance(eye_landmarks[2], eye_landmarks[8])  # 160-144
    vertical_dist3 = euclidean_distance(eye_landmarks[3], eye_landmarks[7])  # 159 - 145
    vertical_dist4 = euclidean_distance(eye_landmarks[4], eye_landmarks[6])  # 158 - 153

    
    # Calculate the Euclidean distance between the set of horizontal eye landmarks
    horizontal_dist = euclidean_distance(eye_landmarks[0], eye_landmarks[5])  # 33 - 133
    
    # Calculate the Eye Aspect Ratio (EAR)
    ear = (vertical_dist1 + vertical_dist2 +vertical_dist3+vertical_dist4) / (4.0 * horizontal_dist)
    return ear


# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Drawing specifications for eye landmarks
draw_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Initialize OpenCV
cap = cv2.VideoCapture(0)
frame_count=0
flag=0
# Start capturing video
with mp_face_mesh.FaceMesh(min_detection_confidence=0.9, min_tracking_confidence=0.9) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB and process it with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Extract face landmarks if available
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks for the left eye 
                left_eye_indices = [362, 384,385, 386,387, 263,373, 374, 380,381]
                left_eye_landmarks = []
                for i in left_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    left_eye_landmarks.append((x, y))
                
                # Extract landmarks for the right eye
                right_eye_indices = [33, 161,160,159, 158, 133, 153, 145,144,163]
                right_eye_landmarks = []
                for i in right_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    right_eye_landmarks.append((x, y))

                left_ear = eye_aspect_ratio(left_eye_landmarks)
                right_ear = eye_aspect_ratio(right_eye_landmarks)
                ear=(left_ear+right_ear )/2.0

                # Draw eye landmarks on the frame
                for (x, y) in left_eye_landmarks + right_eye_landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                if ear <=0.25:
                    flag+=1
                    if flag >= 5:
                        print("Drowsiness")
                else:
                    flag=0
                # Display the EAR values on the frame
                cv2.putText(frame, f'Left EAR: {left_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Average EAR: {(left_ear+right_ear)/2.0:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Right EAR: {right_ear:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(f"saved_frames/frame_{frame_count}.png",frame)
        frame_count+=1
        # Display the frame
        cv2.imshow('Eye Aspect Ratio', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
