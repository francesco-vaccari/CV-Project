import cv2

class FrameDifferencing:
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.prev_frame = None
    
    def apply(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return frame * 0
        
        diff = cv2.absdiff(frame, self.prev_frame)
        diff[diff < self.threshold] = 0
        diff[diff >= self.threshold] = 255

        self.prev_frame = frame
        
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        return diff

class BackgroundSubtractor:
    def __init__(self, bg_path, threshold=50):
        self.bg = cv2.imread(bg_path)
        self.threshold = threshold
    
    def apply(self, frame):
        diff = cv2.absdiff(frame, self.bg)
        diff[diff < self.threshold] = 0
        diff[diff >= self.threshold] = 255

        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        return diff

class AdaptiveBackgroundSubtractor:
    def __init__(self, bg_path, alpha=0.01):
        self.bg = cv2.imread(bg_path)
        self.alpha = alpha

    def apply(self, frame):
        diff = cv2.absdiff(frame, self.bg)
        diff[diff < 50] = 0
        diff[diff >= 50] = 255

        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        self.bg = cv2.addWeighted(frame, self.alpha, self.bg, 1 - self.alpha, 0)

        return diff

def preprocess(mask):
    # preprocessing before extracting bounding boxes
    mask = cv2.medianBlur(mask, 11) # 13 seems a good balance between recall and accuracy, 11 is better with BGSUB
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=18)
    mask = cv2.erode(mask, None, iterations=2)

    # mask = cv2.GaussianBlur(mask, (5,5),0) #blurring appears to reduce accuracy
    return mask

def extract_boxes(mask):
    # extract bounding boxes from mask
    # returns a list of bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes