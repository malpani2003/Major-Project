import os
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# import electrocardiogram 
import matplotlib.pyplot as plt 
from scipy.datasets import electrocardiogram 

# import numpy 
import numpy as np 


# Ensure the directory for saving faces exists
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

start = time.time()

# Ensure the directory for saving frames exists
if not os.path.exists("saved_frames"):
    os.makedirs("saved_frames")

# Initialize the mixer and load the alert sound
mixer.init()
mixer.music.load("music.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    D = distance.euclidean(mouth[0], mouth[4])
    return (A + B + C) / (2.0 * D)


# Thresholds and parameters for drowsiness detection
thresh = 0.20
thresh2 = 0.60
frame_check = 5

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_68_IDXS["inner_mouth"]

ear_dict={"time":[],"ear":[]}
mar_dict={"time":[],"mar":[]}


# def read_ecg():
#     # Simulated ECG data, replace this with your method of reading real ECG data
#     ecg_data = np.random.normal(0, 1, size=(1000,))  # Replace with actual ECG data
#     return ecg_data

# def analyze_ecg(ecg_data):
#     # Analyze ECG data to detect drowsiness
#     # Example: If ECG data indicates irregular heartbeats or low HRV, return True for drowsiness
#     # Replace this with your actual ECG analysis algorithm
#     if np.mean(ecg_data) < 0.1:  # Example condition, replace with actual analysis
#         return True
#     return False



# # define electrocardiogram as ecg model 
# ecg = electrocardiogram() 

# # frequency is 360 
# frequency = 360

# # calculating time data with ecg size along with frequency 
# time_data = np.arange(ecg.size) / frequency 

# # plotting time and ecg model 
# plt.plot(time_data, ecg) 
# plt.xlabel("time in seconds") 
# plt.ylabel("ECG in milli Volts") 
# plt.xlim(9, 10.2) 
# plt.ylim(-1, 1.5) 
# # display 
# plt.show() 



# Start the video stream
cap = cv2.VideoCapture(0)
flag = 0
image_count = 0  # Counter to keep track of the number of saved images
start=time.time()
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Save the frame for debugging purposes
    cv2.imwrite(f"saved_frames/frame_{image_count}.png", gray)

    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)
    # print(len(subjects))

    # ecg=read_ecg()
    # print(ecg)
    # ecg_drowsy=analyze_ecg(ecg)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mstart:mend]

        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthEAR = mouth_aspect_ratio(mouth)

        # Average the EAR
        ear = (leftEAR + rightEAR) / 2.0
        ear_dict["ear"].append(ear)
        ear_dict["time"].append(time.time()-start)

        mar_dict["mar"].append(mouthEAR)
        mar_dict["time"].append(time.time()-start)


        # Visualize the eye landmarks
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Extract the face region from the grayscale image
        (x, y, w, h) = face_utils.rect_to_bb(subject)
        face = gray[y : y + h, x : x + w]

        # Save the grayscale image of the detected face
        cv2.imwrite(f"saved_faces/face_{image_count}.png", face)

        # Check if EAR is below the threshold, if so, increment the flag
        if (ear < thresh and mouthEAR >= thresh2):
            flag += 1
            if flag >= frame_check:
                cv2.putText(
                    frame,
                    "****************ALERT!****************",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "****************ALERT!****************",
                    (10, 325),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                mixer.music.play()
        else:
            flag = 0

    # Display the frame with the eye aspect ratio
    cv2.imshow("Frame", frame)

    # Save the frame with landmarks for debugging purposes
    cv2.imwrite(f"saved_faces/face_landmarks_{image_count}.png", frame)
    image_count += 1

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Plot the EAR over time
plt.figure()
plt.plot(ear_dict["time"], ear_dict["ear"], label="EAR")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.title("Eye Aspect Ratio over Time")
plt.legend()
plt.show()

# Plot the MAR over time
plt.figure()
plt.plot(mar_dict["time"], mar_dict["mar"], label="MAR")
plt.xlabel("Time (s)")
plt.ylabel("MAR")
plt.title("Mouth Aspect Ratio over Time")
plt.legend()
plt.show()