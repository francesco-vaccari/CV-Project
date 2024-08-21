import cv2
import numpy as np
from Annotation import Annotation
from Tracking import OpenCVTracker, DenseOpticalFlowTracker, PyrLKOpticalFlowTracker

video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'

# it may be worth changing the color space of frames from rgb to hsv and remove the brightness component to improve accuracy



cap = cv2.VideoCapture(video)
annotations = Annotation(annotations_folder)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

ret, frame = cap.read()
annotation = annotations.get_annotation(0)

initial_boxes = []
for i in range(len(annotation)):
    initial_boxes.append((min(annotation[i].x1, annotation[i].x2), min(annotation[i].y1, annotation[i].y2), abs(annotation[i].x1 - annotation[i].x2), abs(annotation[i].y1 - annotation[i].y2)))


# tracker = OpenCVTracker('CSRT', frame, initial_boxes)
tracker = DenseOpticalFlowTracker('DISOpticalFlow', frame, initial_boxes, points_sampling='gaussian25')
#tracker = PyrLKOpticalFlowTracker(frame, initial_boxes)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result, boxes = tracker.update(frame)
    
    for i in range(len(result)):
        if result[i]:
            x, y, w, h = [int(i) for i in boxes[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()