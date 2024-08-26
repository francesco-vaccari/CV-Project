import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np



video = 'videos/refined2_short.mp4'
annotation = 'annotations_ball.label'

max_yolo_imgsz = None

# yolo = YOLO("weights/best.pt")
# yolo_class = 0

yolo = YOLO('yolov8x')
yolo_class = 32

decrease_size_factor = 0.8
increase_size_factor = 1.2

min_crop_size = 160





counter = 0
annotation = np.loadtxt(annotation, delimiter=',', dtype=int)
def get_next_annotation():
    global counter, annotation
    print(f'Progress: {counter+1}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
    _, ann_x, ann_y, ann_r = annotation[counter] # center position and radius
    ann_x1 = ann_x - ann_r
    ann_y1 = ann_y - ann_r
    ann_x2 = ann_x + ann_r
    ann_y2 = ann_y + ann_r
    counter += 1
    return ann_x1, ann_y1, ann_x2, ann_y2



def get_yolo_predictions(frame, size):
    image = Image.fromarray(frame)
    results = yolo.predict(image, conf=0.01, classes=[yolo_class], imgsz=size)[0]
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
        if search_x2 - search_x1 < min_crop_size or search_y2 - search_y1 < min_crop_size:
            search_x1 = max(search_x1 - min_crop_size // 2, 0)
            search_y1 = max(search_y1 - min_crop_size // 2, 0)
            search_x2 = min(search_x2 + min_crop_size // 2, frame.shape[1])
            search_y2 = min(search_y2 + min_crop_size // 2, frame.shape[0])     
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


# yolov8x:
# TP: 31, FP: 323, FN: 329
# Precision: 0.08757062146892655, Recall: 0.08611111111111111, F1: 0.08683473389355742

# yolov8m:
# TP: 28, FP: 485, FN: 291
# Precision: 0.05458089668615984, Recall: 0.0877742946708464, F1: 0.0673076923076923

# yolov8n:
# TP: 46, FP: 302, FN: 306
# Precision: 0.13218390804597702, Recall: 0.13068181818181818, F1: 0.13142857142857142

# best.pt @ full size (about 3872):
# TP: 4, FP: 774, FN: 253
# Precision: 0.005141388174807198, Recall: 0.01556420233463035, F1: 0.007729468599033817

# best.pt @ 3200:
# TP: 19, FP: 1233, FN: 104
# Precision: 0.015175718849840255, Recall: 0.15447154471544716, F1: 0.027636363636363633

# best.pt @ 2560:
# TP: 23, FP: 1261, FN: 98
# Precision: 0.01791277258566978, Recall: 0.19008264462809918, F1: 0.03274021352313167

# best.pt @ 2240:
# TP: 22, FP: 1254, FN: 103
# Precision: 0.017241379310344827, Recall: 0.176, F1: 0.03140613847251963

# best.pt @ 1920:
# TP: 2, FP: 770, FN: 264
# Precision: 0.0025906735751295338, Recall: 0.007518796992481203, F1: 0.003853564547206166

# best.pt @ 1600:
# TP: 23, FP: 1256, FN: 104
# Precision: 0.017982799061767005, Recall: 0.18110236220472442, F1: 0.032716927453769556

# best.pt @ 1280:
# TP: 23, FP: 1283, FN: 104
# Precision: 0.01761102603369066, Recall: 0.18110236220472442, F1: 0.03210048848569435

# best.pt @ 960:
# TP: 18, FP: 1312, FN: 86
# Precision: 0.013533834586466165, Recall: 0.17307692307692307, F1: 0.02510460251046025

# best.pt @ 640:
# TP: 19, FP: 1268, FN: 109
# Precision: 0.014763014763014764, Recall: 0.1484375, F1: 0.026855123674911663


# max possible tp: 454
