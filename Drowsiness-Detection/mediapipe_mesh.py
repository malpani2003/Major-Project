# import mediapipe as mp
# import numpy as np
# import cv2
# import imutils


# facmesh = mp.solutions.face_mesh
# face = facmesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
# draw = mp.solutions.drawing_utils
# drawSpec= draw.DrawingSpec(thickness=1,circle_radius=2)


# cap = cv2.VideoCapture(0)
# # Create a full-screen window to get the screen resolution
# # cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
# # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# # screen_width = int(cv2.getWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN))
# # print(screen_width)
# while True:

# 	_, frm = cap.read()
# 	frm = imutils.resize(frm)
# 	# print(frm.shape)
# 	# break
# 	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
# 	rgb = cv2.bilateralFilter(rgb,5,1,1)
    
# 	op = face.process(rgb)
# 	if op.multi_face_landmarks:
# 		for faceLms in op.multi_face_landmarks:
# 			draw.draw_landmarks(frm, faceLms, facmesh.FACEMESH_CONTOURS,drawSpec)
# 			# for id,lm in enumerate(faceLms.landmark):
# 			# 	# print(lm)
# 			# 	ih,iw,ic=frm.shape
# 			# 	x,y = int(lm.x+iw),int(lm.y+ih)
# 			# 	print(id,x,y)
				
# 	cv2.imshow("window", frm)
	
# 	if cv2.waitKey(1) == 27:
# 		cap.release()
# 		cv2.destroyAllWindows()
# 		break
import cv2
import mediapipe as mp
import math

def eye_aspect_ratio(eye_landmarks):
    # Calculate the Euclidean distances between the sets of vertical eye landmarks
    vertical_dist1 = math.sqrt((eye_landmarks[1][0] - eye_landmarks[5][0])**2 + (eye_landmarks[1][1] - eye_landmarks[5][1])**2)
    vertical_dist2 = math.sqrt((eye_landmarks[2][0] - eye_landmarks[4][0])**2 + (eye_landmarks[2][1] - eye_landmarks[4][1])**2)
    
    # Calculate the Euclidean distance between the set of horizontal eye landmarks
    horizontal_dist = math.sqrt((eye_landmarks[0][0] - eye_landmarks[3][0])**2 + (eye_landmarks[0][1] - eye_landmarks[3][1])**2)
    
    # Calculate the Eye Aspect Ratio (EAR)
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize OpenCV
cap = cv2.VideoCapture(0)
flag=0
# Start capturing video
with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
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
                left_eye_indices = [362,385,386,263,374,380]
                left_eye_landmarks = []
                for i in left_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    left_eye_landmarks.append((x, y))
                    
				# right eye ( 33, 133 ,159,145,158,153)
                right_eye_indices=[33,159,158,133,153,145]
                # right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                right_eye_landmarks = []
                for i in right_eye_indices:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    right_eye_landmarks.append((x, y))

                left_ear = eye_aspect_ratio(left_eye_landmarks)
                right_ear = eye_aspect_ratio(right_eye_landmarks)
                ear=(left_ear+right_ear)/2.0
                if ear <= 0.25:
                    cv2.putText(frame, "Drowsiness Detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("Drowsiness Detected")
                else:
                    flag = 0
                # Draw landmarks on the face
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

                # Display the EAR value on the frame
                cv2.putText(frame, f'Left EAR: {left_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Right EAR: {right_ear:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Eye Aspect Ratio', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
