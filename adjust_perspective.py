import cv2
import numpy as np

video = 'videos/out9_combined.mp4'
save_video = 'videos/out9_adjusted.mp4'

cap = cv2.VideoCapture(video)
writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

alpha_src = 1.
alpha_dst = 1.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    src_points = np.float32([
        [0, 0],
        [width, 0],
        [width * (1-alpha_dst), height],
        [width * (alpha_dst), height]
    ])

    dst_points = np.float32([
        [width * (1-alpha_src), 0],
        [width * (alpha_src), 0],
        [0, height],
        [width, height]
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    frame = cv2.warpPerspective(frame, M, (width, height))

    # writer.write(frame)
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        alpha_src += 0.005
    if key == ord('o'):
        alpha_src -= 0.005
    if key == ord('l'):
        alpha_dst += 0.005
    if key == ord('k'):
        alpha_dst -= 0.005


cap.release()
writer.release()

cv2.destroyAllWindows()