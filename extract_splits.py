import cv2
import tqdm

video = 'videos/out9_cut.mp4'
save_left = 'videos/out9_cut_left.mp4'
save_right = 'videos/out9_cut_right.mp4'
split1 = 896
split2 = 2048
split3 = 4096 - 896

# video = 'videos/out10_cut.mp4'
# save_left = 'videos/out10_cut_left.mp4'
# save_right = 'videos/out10_cut_right.mp4'
# split1 = 1024
# split2 = 2048
# split3 = 4096-1024

# video = 'videos/out11_cut.mp4'
# save_left = 'videos/out11_cut_left.mp4'
# save_right = 'videos/out11_cut_right.mp4'
# split1 = 896
# split2 = 2048
# split3 = 4096-896

cap = cv2.VideoCapture(video)

# Create video writers for each frame
# writer1 = cv2.VideoWriter('frame1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (split1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
writer2 = cv2.VideoWriter(save_left, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (split2 - split1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
writer3 = cv2.VideoWriter(save_right, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (split3 - split2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# writer4 = cv2.VideoWriter('frame4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - split3, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm.tqdm(total=total_frames, desc='Saving video')

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
