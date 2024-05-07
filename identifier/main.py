import os
import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import time

video = cv2.VideoCapture(0)

detector = FaceDetector()

def captureAndSaveFace(img, bbox, face_id):
    if isinstance(bbox, dict) and 'bbox' in bbox:
        x, y, w, h = bbox['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        if h > 0 and w > 0:
            face = img[y:y+h, x:x+w]
            cv2.imwrite(f"known_faces/face_{face_id}.jpg", face)
            print(f"Face registered as ID {face_id}")
            time.sleep(5)
        else:
            print("Invalid bounding box")
    else:
        print("Invalid bounding box")


while True:
    _,img = video.read()
    img,bboxes = detector.findFaces(img,draw=True)

    cv2.imshow('Face ID Identifier',img)
    if cv2.waitKey(1) ==27:
        break