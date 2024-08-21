import cv2
import os

video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'

annotations_files = [os.path.join(annotations_folder, file) for file in os.listdir(annotations_folder) if file.endswith('.label')]

cap = cv2.VideoCapture(video)

annotations = []
for file in annotations_files:
    with open(file, 'r') as file:
        entity_annotations = []
        for line in file:
            values = list(map(int, line.strip().split(',')))
            if len(values) == 5:  #Rectangle annotation (player)
                frame, x1, y1, x2, y2 = values
                entity_annotations.append((frame, "rectangle", x1, y1, x2, y2))
            elif len(values) == 4:  #Circle annotation (ball)
                frame, centerx, centery, radius = values
                entity_annotations.append((frame, "circle", centerx, centery, radius))
        annotations.append(entity_annotations)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    for entity in annotations:
        if counter < len(entity):
            if entity[counter][1] == "rectangle":
                _, _, x1, y1, x2, y2 = entity[counter]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            elif entity[counter][1] == "circle":
                _, _, centerx, centery, radius = entity[counter]
                cv2.circle(frame, (centerx, centery), radius, (0, 0, 255), 4)

    cv2.imshow('frame', frame)

    counter += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
