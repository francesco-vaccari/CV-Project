import cv2
import numpy as np


video = cv2.VideoCapture('videos/out9.mp4')

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    split_1 = 896
    split_2 = 2048
    split_3 = 4096 - split_1

    frame1 = frame[:, :split_1]
    frame2 = frame[:, split_1:split_2]
    frame3 = frame[:, split_2:split_3]
    frame4 = frame[:, split_3:]


    # cv2.imshow('Video 1', frame1)
    cv2.imshow('Video 2', frame2)
    # cv2.imshow('Video 3', frame3)
    # cv2.imshow('Video 4', frame4)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
