import cv2
import numpy as np
from Annotation import Annotation
from Tracking import OpenCVTracker, DenseOpticalFlowTracker, PyrLKOpticalFlowTracker, KalmanFilterTracker
import tqdm

video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'
show = False



predicted_tracks = []
predicted_results = []
trackers_order = []

cap = cv2.VideoCapture(video)
annotations = Annotation(annotations_folder)
if show:
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

ret, frame = cap.read()
annotation = annotations.get_annotation(0)

initial_boxes = []
for i in range(len(annotation)):
    initial_boxes.append((min(annotation[i].x1, annotation[i].x2), min(annotation[i].y1, annotation[i].y2), abs(annotation[i].x1 - annotation[i].x2), abs(annotation[i].y1 - annotation[i].y2)))
    trackers_order.append([annotation[i].team, annotation[i].player])

predicted_tracks.append(initial_boxes)
predicted_results.append([True]*len(initial_boxes))

# available trackers for OpenCVTracker: ["CSRT", "MIL", "KCF", "DaSiamRPN", "GOTURN", "Nano", "Vit"] GOTURN is really really slow
# available trackers for DenseOpticalFlowTracker: ["DISOpticalFlow", "FarnebackOpticalFlow"]-
# PyrLKOpticalFlowTracker uses cv2.SparsePyrLKOpticalFlow
# KalmanFilterTracker uses cv2.KalmanFilter

#tracker = OpenCVTracker('CSRT', frame, initial_boxes, show)
# tracker = DenseOpticalFlowTracker('DISOpticalFlow', frame, initial_boxes, points_sampling="gaussian25", show=show)
#tracker = PyrLKOpticalFlowTracker(frame, initial_boxes, points_sampling="gaussian25", show=show)
tracker = KalmanFilterTracker(frame, initial_boxes, show)

progress_bar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
progress_bar.update(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result, boxes = tracker.update(frame)
    predicted_tracks.append(boxes)
    predicted_results.append(result)

    if show:
        for i in range(len(result)):
            if result[i]:
                x, y, w, h = [int(i) for i in boxes[i]]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        cv2.imshow('frame', frame)

    progress_bar.update(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
progress_bar.close()

mean_precision, mean_recall, mean_f1, id_switches, mota, motp = annotations.evaluate_tracking2(predicted_results, predicted_tracks, trackers_order)
print("Average Precision: ", mean_precision)
print("Average Recall: ", mean_recall)
print("Average F1: ", mean_f1)
print("ID Switches: ", id_switches)
print("MOTA: ", mota) # ranges from -inf to 1, higher is better
print("MOTP: ", motp) # ranges from 0 to 1, higher is better
