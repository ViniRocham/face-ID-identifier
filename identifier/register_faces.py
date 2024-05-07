import os
import cv2
from cvzone.FaceDetectionModule import FaceDetector
import time

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

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error opening camera. Make sure the camera is connected properly.")
    exit()

detector = FaceDetector()

face_id = 0

while True:
    ret, img = video.read()
    if not ret:
        print("Error capturing image from camera.")
        break

    img, bboxes = detector.findFaces(img, draw=True)

    cv2.imshow('Face Registration', img)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c'):
        if bboxes:
            for bbox in bboxes:
                captureAndSaveFace(img, bbox, f"face_{face_id}")
                face_id += 1

video.release()
cv2.destroyAllWindows()
