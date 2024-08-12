import cv2
from tqdm import tqdm

video_path = 'videos/out10_transformed.mp4'
save_video = 'videos/out10_transformed_fps_adjusted.mp4'
target_video_framerate = 'videos/out9_transformed.mp4'

cap = cv2.VideoCapture(video_path)

target_fps = cv2.VideoCapture(target_video_framerate).get(cv2.CAP_PROP_FPS)
new_frame_count = cv2.VideoCapture(target_video_framerate).get(cv2.CAP_PROP_FRAME_COUNT)
print(f'Target FPS: {target_fps}')
print(f'New Frame Count: {new_frame_count}')

writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

with tqdm(total=int(new_frame_count)) as pbar:
    for i in range(int(new_frame_count)):
        
        ret, frame = cap.read()
        if not ret:
            break
        
        writer.write(frame)
        pbar.update(1)

cap.release()
writer.release()
