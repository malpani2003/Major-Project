import os
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time

if not os.path.exists("saved_faces_dlib"):
    os.makedirs("saved_faces_dlib")


if not os.path.exists("saved_frames"):
    os.makedirs("saved_frames")

mixer.init()
mixer.music.load("music.wav")


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


# Open a connection to the webcam
cap = cv2.VideoCapture(0)
framenumber=0
start = time.time()
result=0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detect(gray)

    for face in faces:
        # Get the landmarks
        landmarks = predict(gray, face)

        for i in range(68):  # There are 68 landmark points
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imwrite(f"saved_frames/img_{framenumber}.png",frame)
    
    framenumber+=1
    
    cv2.imshow('Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()