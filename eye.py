import cv2
import numpy as np
import os
import pickle
import winsound
from pynput.keyboard import Listener, Key

# creates img directory
try:
    os.makedirs("img")
except:
    pass

# imports cascades
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

#initialises camera
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

i = 0
images = 1000

def test():
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for i in range(len(faces)):
            cv2.rectangle(img, (faces[i][0], faces[i][1]), (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3]), (0, 0, 255), 2)
            face = gray[faces[i][1]:faces[i][1] + faces[i][3], faces[i][0]:faces[i][0] + faces[i][2]]

            eyes = eye_cascade.detectMultiScale(face, 1.3, 5)
            for j in range(len(eyes)):
                cv2.rectangle(img, (eyes[j][0] + faces[i][0], eyes[j][1] + faces[i][1]), (eyes[j][0] + eyes[j][2] + faces[i][0], eyes[j][1] + eyes[j][3] + faces[i][1]), (0, 0, 255), 2)
                eye = img[eyes[j][1] + faces[i][1]:eyes[j][1] + eyes[j][3] + faces[i][1], eyes[j][0] + faces[i][0]:eyes[j][0] + eyes[j][2] + faces[i][0]]

        cv2.imshow("img", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            return

def capture():
    j = 0
    while True:
        # takes picture
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # finds faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            face = gray[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]

            # finds eyes
            eyes = eye_cascade.detectMultiScale(face, 1.3, 5)[:2]

            # finds right eye
            if len(eyes) == 2:
                e = 0
                if eyes[0][0] < eyes[0][1]:
                    e = 1

                eye = gray[eyes[e][1] + faces[0][1]:eyes[e][1] + eyes[e][3] + faces[0][1], eyes[e][0] + faces[0][0]:eyes[e][0] + eyes[e][2] + faces[0][0]]
                eye = cv2.resize(eye, (100, 100), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(f"img/{i}/{j}.png", eye)

                j += 1
                if j > images - 1:
                    return

def on_press(key):
    global i
    if key == Key.space:
        winsound.Beep(1000, 100)
        try:
            os.mkdir(f"img/{i}")
        except:
            pass
        capture()
        winsound.Beep(1000, 100)
        i += 1

def on_release(key):
    if key == Key.space:
        # Stop listener
        #return False
	    pass

def lis():
    # Collect events until released
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

#lis()
test()