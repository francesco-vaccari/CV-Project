import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np



video = 'videos/refined2_short.mp4'
annotation = 'annotations_ball.label'

max_yolo_imgsz = 1280
# yolo = YOLO("weights/best.pt")
yolo = YOLO('yolov8n')

decrease_size_factor = 0.8
increase_size_factor = 1.2





counter = 0
annotation = np.loadtxt(annotation, delimiter=',', dtype=int)
def get_next_annotation():
    global counter, annotation
    _, ann_x, ann_y, ann_r = annotation[counter] # center position and radius
    ann_x1 = ann_x - ann_r
    ann_y1 = ann_y - ann_r
    ann_x2 = ann_x + ann_r
    ann_y2 = ann_y + ann_r
    counter += 1
    return ann_x1, ann_y1, ann_x2, ann_y2



def get_yolo_predictions(frame, size):
    image = Image.fromarray(frame)
    results = yolo.predict(image, conf=0.01, classes=[32], imgsz=size)[0]
    conf = results.boxes.conf.tolist()
    boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in results.boxes.xyxy.tolist()]
    return boxes, conf


def get_imgsz(x, y):
    global max_yolo_imgsz
    size_x = x - x % 32
    size_y = y - y % 32
    if max_yolo_imgsz is None:
        return int(max(size_x, size_y))
    return min(int(max(size_x, size_y)), max_yolo_imgsz)


def evaluate(box):
    pred_x1, pred_y1, pred_x2, pred_y2 = box
    ann_x1, ann_y1, ann_x2, ann_y2 = get_next_annotation()

    if ann_x1 == 0 and ann_y1 == 0 and ann_x2 == 0 and ann_y2 == 0:
        # no annotation
        if pred_x1 == 0 and pred_y1 == 0 and pred_x2 == 0 and pred_y2 == 0:
            # true negative
            return 0, 0, 0
        else:
            # false positive
            return 0, 1, 0
    else:
        # there is annotation
        if pred_x1 == 0 and pred_y1 == 0 and pred_x2 == 0 and pred_y2 == 0:
            # false negative
            return 0, 0, 1
        else:
            # there is prediction
            if (pred_x1 > ann_x1 and pred_x1 < ann_x2 and pred_y1 > ann_y1 and pred_y1 < ann_y2) or \
                (pred_x2 > ann_x1 and pred_x2 < ann_x2 and pred_y2 > ann_y1 and pred_y2 < ann_y2) or \
                (ann_x1 > pred_x1 and ann_x1 < pred_x2 and ann_y1 > pred_y1 and ann_y1 < pred_y2) or \
                (ann_x2 > pred_x1 and ann_x2 < pred_x2 and ann_y2 > pred_y1 and ann_y2 < pred_y2):
                # true positive
                return 1, 0, 0
            else:
                # false positive
                return 0, 1, 0



cap = cv2.VideoCapture(video)
search_x1 = 0
search_y1 = 0
search_x2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
search_y2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tp = 0
fp = 0
fn = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    crop = frame[search_y1:search_y2, search_x1:search_x2]

    imgsz = get_imgsz(crop.shape[1], crop.shape[0])
    boxes, conf = get_yolo_predictions(crop, imgsz)
    cv2.rectangle(frame, (search_x1, search_y1), (search_x2, search_y2), (255, 0, 0), 8)
    
    if len(boxes) > 0:
        pred_x1, pred_y1, pred_x2, pred_y2 = boxes[0]
        cv2.rectangle(frame, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 0), 4)

        search_x1 = min(max(int(search_x1 * (2 - decrease_size_factor)), 0), frame.shape[1])
        search_y1 = min(max(int(search_y1 * (2 - decrease_size_factor)), 0), frame.shape[0])
        search_x2 = max(min(int(search_x2 * (decrease_size_factor)), frame.shape[1]), 0)
        search_y2 = max(min(int(search_y2 * (decrease_size_factor)), frame.shape[0]), 0)
        pred_center = ((pred_x1 + pred_x2) // 2, (pred_y1 + pred_y2) // 2)
        current_center = ((search_x1 + search_x2) // 2, (search_y1 + search_y2) // 2)
        adjustment = (pred_center[0] - current_center[0], pred_center[1] - current_center[1])
        search_x1 += adjustment[0]
        search_x2 += adjustment[0]
        search_y1 += adjustment[1]
        search_y2 += adjustment[1]
        search_x1 = min(max(search_x1, 0), frame.shape[1])
        search_x2 = min(max(search_x2, 0), frame.shape[1])
        search_y1 = min(max(search_y1, 0), frame.shape[0])
        search_y2 = min(max(search_y2, 0), frame.shape[0])        
    else:
        pred_x1, pred_y1, pred_x2, pred_y2 = 0, 0, 0, 0
        search_x1 = min(max(int(search_x1 * (2 - increase_size_factor)), 0), frame.shape[1])
        search_y1 = min(max(int(search_y1 * (2 - increase_size_factor)), 0), frame.shape[0])
        search_x2 = max(min(int(search_x2 * (increase_size_factor)), frame.shape[1]), 0)
        search_y2 = max(min(int(search_y2 * (increase_size_factor)), frame.shape[0]), 0)
    
    is_tp, is_fp, is_fn = evaluate((pred_x1, pred_y1, pred_x2, pred_y2))
    tp += is_tp
    fp += is_fp
    fn += is_fn


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f'TP: {tp}, FP: {fp}, FN: {fn}')
print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')