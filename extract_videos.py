import cv2



# video_path = 'videos/top.mp4' # out9.mp4
# video_to_save_left = 'videos/top_left.mp4'
# video_to_save_right = 'videos/top_right.mp4'
# split1 = 896
# split2 = 2048
# split3 = 4096 - split1

# video_path = 'videos/center.mp4' # out10.mp4
# video_to_save_left = 'videos/center_left.mp4'
# video_to_save_right = 'videos/center_right.mp4'
# split1 = 1026
# split2 = 2048
# split3 = 4096 - split1

# video_path = 'videos/bottom.mp4' # out11.mp4
# video_to_save_left = 'videos/bottom_left.mp4'
# video_to_save_right = 'videos/bottom_right.mp4'
# split1 = 896
# split2 = 2048
# split3 = 4096 - split1

video_path = 'videos/calibration_videos/top.mp4' # out9safe.mp4
video_to_save_left = 'videos/calibration_videos/top_left.mp4'
video_to_save_right = 'videos/calibration_videos/top_right.mp4'
split1 = 896
split2 = 2048
split3 = 4096 - split1

# video_path = 'videos/calibration_videos/center.mp4' # out10safe.mp4
# video_to_save_left = 'videos/calibration_videos/center_left.mp4' 
# video_to_save_right = 'videos/calibration_videos/center_right.mp4'
# split1 = 1026
# split2 = 2048
# split3 = 4096 - split1

# video_path = 'videos/calibration_videos/bottom.mp4' # out11safe.mp4
# video_to_save_left = 'videos/calibration_videos/bottom_left.mp4'
# video_to_save_right = 'videos/calibration_videos/bottom_right.mp4'
# split1 = 896
# split2 = 2048
# split3 = 4096 - split1



video = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter objects for frame2 and frame3
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_frame2 = cv2.VideoWriter(video_to_save_left, fourcc, fps, (split2 - split1, frame_height))
out_frame3 = cv2.VideoWriter(video_to_save_right, fourcc, fps, (split3 - split2, frame_height))

frame_number = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # frame1 = frame[:, :split1]
    frame2 = frame[:, split1:split2]
    frame3 = frame[:, split2:split3]
    # frame4 = frame[:, split3:]

    # Write the frames to the respective video files
    out_frame2.write(frame2)
    out_frame3.write(frame3)

    frame_number += 1
    progress = (frame_number / total_frames) * 100
    print(f"Processing frame {frame_number}/{total_frames} ({progress:.2f}%)", end='\r')

    # Display the frames
    # cv2.imshow('Frame1', frame1)
    # cv2.imshow('Frame2', frame2)
    # cv2.imshow('Frame3', frame3)
    # cv2.imshow('Frame4', frame4)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release everything when the job is finished
video.release()
out_frame2.release()
out_frame3.release()
cv2.destroyAllWindows()