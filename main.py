import cv2
import Detection
from Annotation import Annotation
from yolov5_test import YOLOv5
from yolov8_test import YOLOv8
import tqdm
import matplotlib.pyplot as plt
import argparse


video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'

FD = Detection.FrameDifferencing(threshold=50)
BGSUB = Detection.BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)
ABGSUB = Detection.AdaptiveBackgroundSubtractor(bg_path='background_image.jpg', alpha=0.01)
KNN = cv2.createBackgroundSubtractorKNN()
MOG2 = cv2.createBackgroundSubtractorMOG2()

detector = MOG2
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
show_threshold = 0.4






cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(video)
annotations = Annotation(annotations_folder)

metrics = []
for i in range(len(threshold_values)):
    metrics.append([])

progress_bar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    mask = detector.apply(frame)

    mask = Detection.preprocess(mask)

    boxes = Detection.extract_boxes(mask)

    correct_box_indexes = []
    for i, threshold in enumerate(threshold_values):
        precision, recall, indexes = annotations.evaluate(boxes, int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1, threshold=threshold)
        metrics[i].append((precision, recall))
        if threshold == show_threshold:
            correct_box_indexes = indexes

    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if j in correct_box_indexes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    
    progress_bar.update(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
progress_bar.close()


average_precisions = []
average_recalls = []
for i, threshold in enumerate(threshold_values):
    precisions = []
    recalls = []

    for metric in metrics[i]:
        precisions.append(metric[0])
        recalls.append(metric[1])
    
    precision = sum(precisions)/len(precisions)
    recall = sum(recalls)/len(recalls)

    average_precisions.append(precision)
    average_recalls.append(recall)
    
    print('----------------------')
    print(f'IoU Threshold: {threshold}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print('----------------------')

# mAP over IoU thresholds
mAP = sum(average_precisions)/len(average_precisions)
print(f'mAP: {mAP}')
print('----------------------')










# yolo = YOLOv8()
# yolo = YOLOv5()
# bboxes = yolo.predict(frame, imgsz=640*5)
# bboxes = yolo.predict(frame, imgsz=640)