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


def loadKnownFaces():
    known_faces = {}
    for filename in os.listdir("known_faces"):
        name = os.path.splitext(filename)[0]
        known_faces[name] = cv2.imread(os.path.join("known_faces", filename), cv2.IMREAD_GRAYSCALE)
    return known_faces


def identifyFace(face, known_faces):
    pass


video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error opening camera. Make sure the camera is connected properly.")
    exit()

detector = FaceDetector()

known_faces = loadKnownFaces()
face_id = 0

while True:
    ret, img = video.read()
    if not ret:
        print("Error capturing image from camera.")
        break

    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        for bbox in bboxes:
            try:
                face = cv2.cvtColor(img[bbox['bbox'][1]:bbox['bbox'][3], bbox['bbox'][0]:bbox['bbox'][2]], cv2.COLOR_BGR2GRAY)
                name = identifyFace(face, known_faces)
                cv2.putText(img, name, (bbox['bbox'][0], bbox['bbox'][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error processing face: {e}")

    cv2.imshow('Face ID Identifier', img)

    key = cv2.waitKey(1)