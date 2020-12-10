import cv2
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from pynput.mouse import Controller as mc

from camera import *

mouse = mc()

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

input_size = 100 ** 2
num_classes = 4
model = NN(input_size, num_classes)

# Load
model.load_state_dict(torch.load("model.pt"))

def vid():
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

                cv2.imshow("img", eye)
                cv2.waitKey(33)

                tens = torch.tensor(eye).reshape(-1).float()

                with torch.no_grad():
                    p = model(tens)
                    r = p.max(0).indices

                    if r == 0:
                        mouse.position = (468, 258)
                    elif r == 1:
                        mouse.position = (1428, 258)
                    elif r == 2:
                        mouse.position = (468, 798)
                    elif r == 3:
                        mouse.position = (1428, 798)

vid()