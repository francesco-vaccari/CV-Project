import os
import cv2

annotations_folder = 'annotations'
reference_video = 'videos/refined2_short.mp4'



cap = cv2.VideoCapture(reference_video)
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

annotations_files = [os.path.join(annotations_folder, file) for file in os.listdir(annotations_folder) if file.endswith('.txt')]

for file_path in annotations_files:
    out = []
    with open(file_path, 'r') as file:
        for line in file:
            frame, x1, y1, x2, y2 = line.strip().split(',')
            frame, x1, y1, x2, y2 = int(frame), int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            out.append(f'{frame},{x1},{y1},{x2},{y2}')
    out_path = file_path[:-4] + '.label'
    with open(out_path, 'w') as file:
        for line in out:
            file.write(line + '\n')

cap.release()