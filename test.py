import cv2
import numpy as np
import os
import pickle
import winsound
from pynput.keyboard import Listener, Key

from camera import *

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

test()