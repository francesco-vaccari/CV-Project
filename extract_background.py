import cv2
import numpy as np


video_path = 'videos/refined2_short.mp4'



is_paused = False
background_image = None
current_frame = None
roi_points = []
drawing = False

def refine_background():
    global background_image, current_frame
    background_image = np.where(background_image == 0, current_frame, background_image)

def mouse_callback(event, x, y, flags, param):
    global roi_points, drawing, current_frame, background_image

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = current_frame.copy()
            cv2.rectangle(temp_frame, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('Video', temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        drawing = False

        if len(roi_points) == 2:
            x1, y1 = roi_points[0]
            x2, y2 = roi_points[1]

            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            roi = current_frame[y1:y2, x1:x2]

            cv2.imshow('Video', current_frame)

            if background_image is None:
                background_image = np.zeros_like(current_frame)

            background_image[y1:y2, x1:x2] = roi

            cv2.imshow('Background Image', background_image)

            roi_points = []


cap = cv2.VideoCapture(video_path)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video', mouse_callback)

while True:
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = frame.copy()
        cv2.imshow('Video', frame)

    key = cv2.waitKey(30) & 0xFF

    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if key == ord('a'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - cap.get(cv2.CAP_PROP_FPS) * 3)
    if key == ord('d'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + cap.get(cv2.CAP_PROP_FPS) * 3)
    if key == ord('q'):
        break
    elif key == ord('p'):
        is_paused = not is_paused
    elif key == ord('s') and background_image is not None:
        refine_background()
        cv2.imwrite('background_image.jpg', background_image)
        print('Background image saved')

cap.release()
cv2.destroyAllWindows()


