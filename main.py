import cv2
from Detection import FrameDifferencing, BackgroundSubtractor, AdaptiveBackgroundSubtractor

FD = FrameDifferencing(threshold=50)
BGSUB = BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)
ABGSUB = AdaptiveBackgroundSubtractor()
KNN = cv2.createBackgroundSubtractorKNN()
MOG2 = cv2.createBackgroundSubtractorMOG2()

detector = BGSUB


cap = cv2.VideoCapture('videos/refined.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    mask = detector.apply(frame)

    cv2.imshow('Mask', mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()