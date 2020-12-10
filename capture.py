import cv2
import numpy as np
import os
import pickle
import winsound
from pynput.keyboard import Listener, Key
from playsound import playsound

from camera import *

# creates img directory
try:
    os.makedirs("img")
except:
    pass

i = 0
image_number = 100

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
                if j > image_number - 1:
                    return

def on_press(key):
    global i
    if key == Key.space:
        playsound("sounds/start.mp3")
        try:
            os.mkdir(f"img/{i}")
        except:
            pass
        capture()
        playsound("sounds/stop.mp3")
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

lis()