import torch
import numpy as np
from PIL import Image

class YOLOv5:
    def __init__(self):
        # pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')
        self.yolo.classes = [0]
        self.yolo.max_det = 12

    def computeIntersection(self, fx1, fy1, fx2, fy2, sx1, sy1, sx2, sy2):
        dx = min(fx2, sx2) - max(fx1, sx1)
        dy = min(fy2, sy2) - max(fy1, sy1)
        if (dx>=0) and (dy>=0):
            area = dx*dy
        else:
            area = 0
        return area

    def computeAccuracy(self, bboxes, index, label):
        x_min, y_min, x_max, y_max = bboxes['xmin'][index], bboxes['ymin'][index], bboxes['xmax'][index], bboxes['ymax'][index]
        x, y, w, h = label[0].item(), label[1].item(), label[2].item(), label[3].item()

        intersection = self.computeIntersection(x_min, y_min, x_max, y_max, x, y, x+w, y+h)

        area1 = (x_max - x_min) * (y_max - y_min)
        area2 = w * h

        return intersection / (area1 + area2 - intersection)

    def predict(self, image):
        image = Image.fromarray(image)
        result = self.yolo(image)
        bbox = result.pandas().xyxy[0]
        bbox = bbox.reset_index()
        bbox["tconfidence"] = np.nan
        bbox["crop"] = np.nan

        bboxes = []
        for index, row in bbox.iterrows():
            box = (
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax']),
            )
            bboxes.append(box)
        
        return bboxes