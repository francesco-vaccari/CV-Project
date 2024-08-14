from ultralytics import YOLO

class YOLOv8:
    def __init__(self):
        self.model = YOLO('yolov8x.pt')
        self.classes = [0]
        self.conf_thresh = 0.1
    
    def predict(self, image):
        result = self.model.predict(image, conf=self.conf_thresh, classes=self.classes)
        boxes = result[0].boxes
        bboxes = []
        for i in range(boxes.shape[0]):
            x = int(boxes.xyxy[i, 0])
            y = int(boxes.xyxy[i, 1])
            x2 = int(boxes.xyxy[i, 2])
            y2 = int(boxes.xyxy[i, 3])
            bboxes.append((x, y, x2, y2))
        
        return bboxes
