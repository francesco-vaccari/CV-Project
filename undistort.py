import cv2
import numpy as np
import glob


# Load calibration data
calibration_data = np.load("calibration_data_out9_frame2.npz")
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# Read the video
cap = cv2.VideoCapture('videos/out9_frame2.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('out9_frame2_corrected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Undistort the image
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
    
    # Crop the image based on ROI
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # Write the frame to the output video
    out.write(undistorted_frame)
    
    # cv2.imshow('Undistorted Frame', undistorted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
