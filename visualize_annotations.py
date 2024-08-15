import cv2

video = 'videos/refined2_short.mp4'
annotations_file = 'annotations_1_4.txt'

cap = cv2.VideoCapture(video)

annotations = []
with open(annotations_file, 'r') as file:
    for line in file:
        frame, x1, y1, x2, y2 = line.strip().split(',')
        annotations.append((int(frame), int(x1), int(y1), int(x2), int(y2)))

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.rectangle(frame, (annotations[counter][1], annotations[counter][2]), (annotations[counter][3], annotations[counter][4]), (0, 255, 0), 4)

    cv2.imshow('frame', frame)

    counter += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
