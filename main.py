import cv2
from Detection import FrameDifferencing, BackgroundSubtractor, AdaptiveBackgroundSubtractor

FD = FrameDifferencing(threshold=50)
BGSUB = BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)
ABGSUB = AdaptiveBackgroundSubtractor(bg_path='background_image.jpg', alpha=0.01)
KNN = cv2.createBackgroundSubtractorKNN()
MOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)

detector = MOG2







def get_bounding_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    
    return boxes

cap = cv2.VideoCapture('videos/refined.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    mask = detector.apply(frame)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=18)

    boxes = get_bounding_boxes(mask)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Mask', mask)
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()