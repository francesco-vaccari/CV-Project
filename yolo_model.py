import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

class yolo_model:
    def __init__(self):
        self.yolo = YOLO(model='weights/best.pt')
    
    def predict(self, image):
        result = self.yolo.predict(image, conf=0.25, imgsz=1280, classes=[1], max_det=12)
        boxes = result[0].boxes
        bboxes = []
        for i in range(boxes.shape[0]):
            x = int(boxes.xyxy[i, 0])
            y = int(boxes.xyxy[i, 1])
            x2 = int(boxes.xyxy[i, 2])
            y2 = int(boxes.xyxy[i, 3])
            bboxes.append((x, y, x2, y2))
        
        return bboxes

if __name__ == '__main__':
    import cv2
    
    yolo = yolo_model()
    video = 'videos/refined2_short.mp4'

    cap = cv2.VideoCapture(video)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = yolo.predict(frame)

        for box in bboxes:
            x, y, x2, y2 = box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 4)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
