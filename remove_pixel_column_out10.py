import cv2

video = 'videos/out10_cut_right.mp4'
save_video = 'videos/out10_cut_right_adjusted.mp4'

n = 2

cap = cv2.VideoCapture(video)
writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # cv2.imshow('Image', frame)

    # remove the first column of pixels
    frame = frame[:, n:]
    # cv2.imshow('Image2', frame)

    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()

cv2.destroyAllWindows()