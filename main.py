import cv2
import numpy as np
import argparse

#Scale the video frames
def scale_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height))

#Parse command line arguments
parser = argparse.ArgumentParser(description='Scale the window size')
parser.add_argument('--scale', type=int, default=100, help='Scale percentage of the window size (default: 100)')
args = parser.parse_args()

#Take our 3 videos
top = cv2.VideoCapture('videos/top.mp4')
center = cv2.VideoCapture('videos/center.mp4')
down = cv2.VideoCapture('videos/down.mp4')

while top.isOpened():
    ret, frameTop = top.read()
    ret2, frameCenter = center.read()
    ret3, frameDown = down.read()

    if not (ret and ret2 and ret3):
        break

    #Divide video 1 in 4 frames
    split_1 = 896
    split_2 = 2048
    split_3 = 4096 - split_1

    frame1 = frameTop[:, :split_1]
    frame2 = frameTop[:, split_1:split_2]
    frame3 = frameTop[:, split_2:split_3]
    frame4 = frameTop[:, split_3:]

    #Reshape each frame of the top video
    height, width, _ = frame1.shape
    frame2 = cv2.resize(frame2, (width, height))  
    frame3 = cv2.resize(frame3, (width, height))
    frame4 = cv2.resize(frame4, (width, height))
      
    #Stack frames
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))
    stacked_video = np.hstack((top_row, bottom_row))
    
    #Scale the video
    scaled_frame = scale_frame(stacked_video, args.scale)
    
    #Show and close program
    cv2.imshow('Stacked video', scaled_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

top.release()
center.release()
down.release()
cv2.destroyAllWindows()
