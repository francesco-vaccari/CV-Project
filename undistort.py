import cv2
import numpy as np



input_video_path = 'videos/top_left.mp4'
calibration_data_path = 'videos/calibration_videos/calibration_data_top_left.npz'
output_video_path = 'videos/top_left_corrected.mp4'



# Load camera calibration data
calibration_data = np.load(calibration_data_path)
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

# Open input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    out.write(undistorted_frame)

    frame_number += 1
    progress = (frame_number / total_frames) * 100
    print(f"Processing frame {frame_number}/{total_frames} ({progress:.2f}%)", end='\r')

    # Display the undistorted frame
    # cv2.imshow('Undistorted Video', undistorted_frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
