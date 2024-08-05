import cv2
import tqdm

video = 'out9.mp4'

cap = cv2.VideoCapture(video)
split1 = 896
split2 = 2048
split3 = 4096 - 896

# Create video writers for each frame
# writer1 = cv2.VideoWriter('frame1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (split1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
writer2 = cv2.VideoWriter('frame2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (split2 - split1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
writer3 = cv2.VideoWriter('frame3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (split3 - split2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# writer4 = cv2.VideoWriter('frame4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - split3, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm.tqdm(total=total_frames, desc='Processing Frames')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame1 = frame[:, :split1]
    frame2 = frame[:, split1:split2]
    frame3 = frame[:, split2:split3]
    # frame4 = frame[:, split3:]

    # Write each frame to the corresponding video file
    # writer1.write(frame1)
    writer2.write(frame2)
    writer3.write(frame3)
    # writer4.write(frame4)

    progress_bar.update(1)  # Update progress bar

cap.release()

# Release the video writers
# writer1.release()
writer2.release()
writer3.release()
# writer4.release()

cv2.destroyAllWindows()
