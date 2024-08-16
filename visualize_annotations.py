import cv2
import os

video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'



annotations_files = [os.path.join(annotations_folder, file) for file in os.listdir(annotations_folder) if file.endswith('.label')]

cap = cv2.VideoCapture(video)

annotations = []
for file in annotations_files:
    with open(file, 'r') as file:
        player = []
        for line in file:
            frame, x1, y1, x2, y2 = line.strip().split(',')
            player.append((int(frame), int(x1), int(y1), int(x2), int(y2)))
        annotations.append(player)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    for player in annotations:
        cv2.rectangle(frame, (player[counter][1], player[counter][2]), (player[counter][3], player[counter][4]), (0, 255, 0), 4)

    cv2.imshow('frame', frame)

    counter += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
