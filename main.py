import cv2
import cv2.bgsegm as bgseg
import Detection
from Annotation import Annotation
from yolov5_test import YOLOv5
from yolov8_test import YOLOv8
import tqdm
import matplotlib.pyplot as plt
import argparse


video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'

notes_folder = 'notes/'
notes_name = 'MOG2_thinMedianBlur13'
note = open(notes_folder+notes_name+".txt", "w")

FD = Detection.FrameDifferencing(threshold=50)
BGSUB = Detection.BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)
ABGSUB = Detection.AdaptiveBackgroundSubtractor(bg_path='background_image.jpg', alpha=0.01)
KNN = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
MOG2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
CNT = bgseg.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True, maxPixelStability=1*60, isParallel=True)
GMG = bgseg.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8)
GSOC = bgseg.createBackgroundSubtractorGSOC(mc=bgseg.LSBP_CAMERA_MOTION_COMPENSATION_NONE, nSamples=20, replaceRate=0.003,
                                            propagationRate=0.01, hitsThreshold=32, alpha= 0.01, beta=0.0022,
                                            blinkingSupressionDecay=0.1, blinkingSupressionMultiplier=0.1,
                                            noiseRemovalThresholdFacBG=0.0004, noiseRemovalThresholdFacFG=0.0008)
LSBP = bgseg.createBackgroundSubtractorLSBP(mc=bgseg.LSBP_CAMERA_MOTION_COMPENSATION_NONE, nSamples=20, LSBPRadius=16,
                                            Tlower=2.0, Tupper=32.0, Tinc=1.0, Tdec=0.05, Rscale=10.0, Rincdec=0.005,
                                            noiseRemovalThresholdFacBG=0.0004, noiseRemovalThresholdFacFG=0.0008,
                                            LSBPthreshold=8, minCount=2)


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
    note.write("----------------------\n")
    print(f'IoU Threshold: {threshold}')
    note.write("IoU Threshold:"+str(threshold)+"\n")
    print(f'Precision: {precision}')
    note.write("Precision:"+str(precision)+"\n")
    print(f'Recall: {recall}')
    note.write("Recall:"+str(recall)+"\n")
    print('----------------------')
    note.write("----------------------")

# mAP over IoU thresholds
mAP = sum(average_precisions)/len(average_precisions)
print(f'mAP: {mAP}')
note.write("mAP:"+str(mAP)+"\n")
print('----------------------')
note.write("----------------------")

note.close()










# yolo = YOLOv8()
# yolo = YOLOv5()
# bboxes = yolo.predict(frame, imgsz=640*5)
# bboxes = yolo.predict(frame, imgsz=640)