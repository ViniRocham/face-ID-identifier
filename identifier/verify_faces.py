import os
import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
import time

def loadKnownFaces():
    known_faces = {}
    for filename in os.listdir("known_faces"):
        name = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join("known_faces", filename), cv2.IMREAD_GRAYSCALE)
        known_faces[name] = img
    return known_faces

def identifyFace(face, known_faces):
    if len(face.shape) > 2:
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face

    if face_gray.size == 0:
        return None

    for name, known_face in known_faces.items():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([known_face], np.array([0]))
        label, confidence = recognizer.predict(face_gray)
        
        threshold = 70
        if confidence < threshold:
            return name
    return None

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error opening camera. Make sure the camera is connected properly.")
    exit()

detector = FaceDetector()
known_faces = loadKnownFaces()

print("Loading...")

while True:
    ret, img = video.read()
    if not ret:
        print("Error capturing image from camera.")
        break

    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        for bbox in bboxes:
            try:
                face = img[bbox['bbox'][1]:bbox['bbox'][3], bbox['bbox'][0]:bbox['bbox'][2]]
                name = identifyFace(face, known_faces)
                if name:
                    print(f"{name} is registered.")
                    
                    time.sleep(10)
                    
                    video.release()
                    cv2.destroyAllWindows()
                    exit()
                else:
                    print("Person is not registered.")
                    continue
            except Exception as e:
                pass

            print("\n")
            break

    cv2.imshow('Face Identification', img)

    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
