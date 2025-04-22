import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import winsound

# Initialize Pygame Mixer for sound
mixer.init()
sound = mixer.Sound("alarm.wav")

# Load Haar Cascade files for face, eyes, and mouth
face = cv2.CascadeClassifier("haar cascade files/haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("haar cascade files/haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("haar cascade files/haarcascade_righteye_2splits.xml")
mouth = cv2.CascadeClassifier("haar cascade files/mouth.xml")  # Adjust path if needed

# Load pre-trained model for predicting eye state
model = load_model("models/cnncvip3.h5")
path = os.getcwd()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
yawn_pred = [99]  # For mouth predictions

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face, eyes, and mouth in the frame
    faces = face.detectMultiScale(
        gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25)
    )
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    mouths = mouth.detectMultiScale(
        gray,
        minNeighbors=5,
        scaleFactor=1.1,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Draw a black rectangle at the bottom for displaying text
    cv2.rectangle(
        frame, (0, height - 70), (300, height), (0, 0, 0), thickness=cv2.FILLED
    )

    # Face detection
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Right eye detection and prediction
    for x, y, w, h in right_eye:
        r_eye = frame[y : y + h, x : x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if rpred[0] == 1:
            lbl = "Open"
        if rpred[0] == 0:
            lbl = "Closed"
        break

    # Left eye detection and prediction
    for x, y, w, h in left_eye:
        l_eye = frame[y : y + h, x : x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if lpred[0] == 1:
            lbl = "Open"
        if lpred[0] == 0:
            lbl = "Closed"
        break

    # Mouth detection and prediction
    for x, y, w, h in mouths:
        mouth_img = frame[y : y + h, x : x + w]
        mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
        mouth_img = cv2.resize(mouth_img, (24, 24))
        mouth_img = mouth_img / 255
        mouth_img = mouth_img.reshape(24, 24, -1)
        mouth_img = np.expand_dims(mouth_img, axis=0)
        yawn_pred = np.argmax(
            model.predict(mouth_img), axis=-1
        )  # Predicting mouth status
        if yawn_pred[0] == 3:
            cv2.putText(
                frame,
                "Yawning",
                (10, height - 50),
                font,
                1,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
            score += 2  # Increase score for yawning
        break

    # Update score based on eye state
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1  # Increase score if both eyes are closed
        cv2.putText(
            frame,
            "Eyes Closed",
            (10, height - 30),
            font,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        score -= 1  # Decrease score if any eye is open
        cv2.putText(
            frame,
            "Eyes Open",
            (10, height - 30),
            font,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Ensure score doesn't go below zero
    score = max(score, 0)

    # Display score on the frame
    cv2.putText(
        frame,
        "Score:" + str(score),
        (100, height - 50),
        font,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Sound alarm if score is high
    if score > 20:
        cv2.imwrite(os.path.join(path, "image.jpg"), frame)
        try:
            sound.play()
        except:  # Sound is already playing
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Show the frame
    cv2.imshow("frame", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
