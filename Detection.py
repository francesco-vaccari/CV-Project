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
    def __init__(self):
        pass

    def apply(self, frame):

        pass