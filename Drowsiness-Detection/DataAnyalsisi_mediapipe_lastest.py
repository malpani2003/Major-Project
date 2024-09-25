import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os
import time
import pygame.mixer as mixer
import json
from datetime import datetime

# Initialize the mixer and load the alert sound
mixer.init()
mixer.music.load("music.wav")

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def eye_aspect_ratio(eye_landmarks):
    """Calculate the Eye Aspect Ratio (EAR) to detect drowsiness."""
    vertical_dist1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_dist2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    horizontal_dist = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    """Calculate the Mouth Aspect Ratio (MAR) to detect yawning."""
    vertical_dist1 = euclidean_distance(mouth_landmarks[9], mouth_landmarks[10])
    vertical_dist2 = euclidean_distance(mouth_landmarks[11], mouth_landmarks[12])
    vertical_dist3 = euclidean_distance(mouth_landmarks[13], mouth_landmarks[14])
    horizontal_dist = euclidean_distance(mouth_landmarks[0], mouth_landmarks[1])
    mar = (vertical_dist1 + vertical_dist2 + vertical_dist3) / (3.0 * horizontal_dist)
    return mar

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Drawing specifications for eye landmarks
draw_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Initialize OpenCV
cap = cv2.VideoCapture(0)
frame_count = 0
flag = 0
alert_flag = 0

# Initialize variables for FPS calculation
start_time = time.time()
frame_counter = 0
fps = 0

# Ensure the directory for saved frames exists
os.makedirs('saved_frames_mediapipe', exist_ok=True)

# Latency calculation variables
latency_values = []

# Initialize data storage
users_data = {}
current_user = "default_user"  # You can implement a user login system to set this

def save_user_data(user_id, timestamp, ear, mar, drowsiness_state):
    if user_id not in users_data:
        users_data[user_id] = []
    
    users_data[user_id].append({
        "timestamp": timestamp,
        "ear": ear,
        "mar": mar,
        "drowsiness_state": drowsiness_state
    })

def save_data_to_file():
    with open("drowsiness_data.json", "w") as f:
        json.dump(users_data, f)

# Start capturing video
with mp_face_mesh.FaceMesh(min_detection_confidence=0.9, min_tracking_confidence=0.9) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_frame_time = time.time()

        # Convert the image to RGB and process it with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Extract face landmarks if available
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks for the left eye 
                left_eye_indices = [362, 386, 387, 263, 373, 374]
                left_eye_landmarks = []
                for i in left_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    left_eye_landmarks.append((x, y))
                
                # Extract landmarks for the right eye
                right_eye_indices = [33, 159, 158, 133, 153, 145]
                right_eye_landmarks = []
                for i in right_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    right_eye_landmarks.append((x, y))

                # Extract landmarks for the mouth
                mouth_indices = [78, 308, 191, 95, 80, 88, 81, 178, 82, 87, 13, 14, 312, 317, 311, 402, 310, 318, 415, 324]
                mouth_landmarks = []
                for i in mouth_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    mouth_landmarks.append((x, y))

                left_ear = eye_aspect_ratio(left_eye_landmarks)
                right_ear = eye_aspect_ratio(right_eye_landmarks)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(mouth_landmarks)

                # Draw landmarks on the face
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

                # Determine drowsiness state
                drowsiness_state = "Normal"
                if ear <= 0.25:
                    alert_flag += 1
                    if alert_flag >= 10:
                        drowsiness_state = "Alert"
                        cv2.putText(frame, "Alert", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
                elif ear <= 0.29 or mar >= 0.60:
                    flag += 1
                    if flag >= 10:
                        drowsiness_state = "Warning"
                        cv2.putText(frame, "Warning", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    flag = 0
                    alert_flag = 0
                    mixer.music.stop()

                # Save user data
                timestamp = datetime.now().isoformat()
                save_user_data(current_user, timestamp, ear, mar, drowsiness_state)

                # Display the EAR and MAR values
                cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the frame
        cv2.imwrite(f"saved_frames_mediapipe/frame_{frame_count}.png", frame)
        frame_count += 1

        # Display FPS
        frame_counter += 1
        if time.time() - start_time >= 1:
            fps = frame_counter / (time.time() - start_time)
            frame_counter = 0
            start_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Eye and Mouth Aspect Ratio', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Save all collected data to a file
save_data_to_file()

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

def analyze_user_data(user_id):
    user_data = users_data.get(user_id, [])
    if not user_data:
        print(f"No data available for user {user_id}")
        return

    ears = [entry["ear"] for entry in user_data]
    mars = [entry["mar"] for entry in user_data]
    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in user_data]
    drowsiness_states = [entry["drowsiness_state"] for entry in user_data]

    # Calculate statistics
    avg_ear = sum(ears) / len(ears)
    avg_mar = sum(mars) / len(mars)
    
    # Analyze drowsiness states with start and end times
    drowsiness_periods = []
    current_state = drowsiness_states[0]
    start_time = timestamps[0]
    
    for i in range(1, len(drowsiness_states)):
        if drowsiness_states[i] != current_state:
            drowsiness_periods.append({
                "state": current_state,
                "start_time": start_time,
                "end_time": timestamps[i - 1],
                "duration": timestamps[i - 1] - start_time
            })
            current_state = drowsiness_states[i]
            start_time = timestamps[i]
    
    # Add the last period
    drowsiness_periods.append({
        "state": current_state,
        "start_time": start_time,
        "end_time": timestamps[-1],
        "duration": timestamps[-1] - start_time
    })

    # Calculate total duration for each state
    drowsiness_durations = {}
    for period in drowsiness_periods:
        if period["state"] not in drowsiness_durations:
            drowsiness_durations[period["state"]] = timedelta()
        drowsiness_durations[period["state"]] += period["duration"]

    # Plotting
    plt.figure(figsize=(12, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, ears, label='EAR')
    plt.plot(timestamps, mars, label='MAR')
    plt.title(f'EAR and MAR over time - User {user_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Ratio')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.bar(drowsiness_durations.keys(), [d.total_seconds() / 60 for d in drowsiness_durations.values()])
    plt.title(f'Drowsiness State Distribution - User {user_id}')
    plt.xlabel('State')
    plt.ylabel('Duration (minutes)')

    plt.subplot(3, 1, 3)
    for i, period in enumerate(drowsiness_periods):
        plt.barh(i, (period["end_time"] - period["start_time"]).total_seconds() / 60, 
                 left=(period["start_time"] - timestamps[0]).total_seconds() / 60, 
                 color={'Normal': 'green', 'Warning': 'yellow', 'Alert': 'red'}[period["state"]])
    plt.yticks(range(len(drowsiness_periods)), [f"{p['state']}: {p['start_time'].strftime('%H:%M:%S')} - {p['end_time'].strftime('%H:%M:%S')}" for p in drowsiness_periods])
    plt.title(f'Drowsiness State Timeline - User {user_id}')
    plt.xlabel('Time (minutes)')

    plt.tight_layout()
    plt.savefig(f'user_{user_id}_analysis.png')
    plt.close()

    print(f"Analysis for User {user_id}:")
    print(f"Average EAR: {avg_ear:.2f}")
    print(f"Average MAR: {avg_mar:.2f}")
    print("Drowsiness State Distribution:")
    for state, duration in drowsiness_durations.items():
        print(f"  {state}: {duration.total_seconds() / 60:.2f} minutes")
    print("\nDrowsiness Periods:")
    for period in drowsiness_periods:
        print(f"  {period['state']}: {period['start_time'].strftime('%H:%M:%S')} - {period['end_time'].strftime('%H:%M:%S')} (Duration: {period['duration'].total_seconds() / 60:.2f} minutes)")
        
# Perform analysis for each user
for user_id in users_data.keys():
    analyze_user_data(user_id)

print("Analysis complete. Check the generated PNG files for visualizations.")

# Plot the latency graph
plt.figure()
plt.plot(latency_values)
plt.xlabel("Frame")
plt.ylabel("Latency (s)")
plt.title("Software Latency")
plt.savefig('latency_analysis.png')
plt.close()

print("Latency analysis complete. Check latency_analysis.png for visualization.")