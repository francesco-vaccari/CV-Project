import cv2

video = cv2.VideoCapture('videos/out9.mp4')

# Get the width and height of the video frames
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# change if needed
split1 = 896
split2 = 2048
split3 = 4096 - split1

# Define the codec and create VideoWriter objects for frame2 and frame3
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_frame2 = cv2.VideoWriter('videos/out_frame2.mp4', fourcc, fps, (split2 - split1, frame_height))
out_frame3 = cv2.VideoWriter('videos/out_frame3.mp4', fourcc, fps, (split3 - split2, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame2 = frame[:, split1:split2]
    frame3 = frame[:, split2:split3]

    # Write the frames to the respective video files
    out_frame2.write(frame2)
    out_frame3.write(frame3)

# Release everything when the job is finished
video.release()
out_frame2.release()
out_frame3.release()
cv2.destroyAllWindows()