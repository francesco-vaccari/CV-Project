import cv2
import numpy as np
from tqdm import tqdm

calibration_file = 'videos/out9safe_right_frames.npz'
video_to_calibrate = 'videos/out9_right.mp4'
name_video_output = 'videos/out9_right_undistorted.mp4'

# Load the calibration file
calibration = np.load(calibration_file)
mtx = calibration['mtx']
dist = calibration['dist']

# Load the video
cap = cv2.VideoCapture(video_to_calibrate)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a VideoWriter object (mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name_video_output, fourcc, fps, (width, height))

# Create a progress bar
progress_bar = tqdm(total=total_frames, desc='Processing Frames', unit='frame')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame
    undistorted = cv2.undistort(frame, mtx, dist, None)

    out.write(undistorted)

    # Update the progress bar
    progress_bar.update(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()