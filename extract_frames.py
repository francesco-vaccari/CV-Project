import cv2
import os

# Path to the video file
video_path = 'videos/out9safe_frame2.mp4'
# Directory to save the extracted frames
frames_dir = 'out9safe_frame2'

# Create the directory if it does not exist
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Capture the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
saved_frame_count = 0
last_saved_frame = -15  # Initialize to ensure the first frame can be saved

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If spacebar is pressed and at least 15 frames have passed since the last save
    if key == ord(' ') and (frame_count - last_saved_frame >= 15):
        frame_filename = os.path.join(frames_dir, f'frame_{saved_frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1
        last_saved_frame = frame_count
        print(f'Saved frame {saved_frame_count} as {frame_filename}')

    # If 'q' is pressed, exit the loop
    if key == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Extracted {saved_frame_count} frames.")
