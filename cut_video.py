import cv2

video = 'videos/refined2.mp4'
save_video = 'videos/refined2_short.mp4'

start_second = 15
end_second = 90

cap = cv2.VideoCapture(video)
framerate = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_second * framerate)
end_frame = int(end_second * framerate)
print(f'Start frame {start_frame}, end frame {end_frame}')

out = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), framerate, (int(cap.get(3)), int(cap.get(4))))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f'Saving frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}/{total_frames}', end='\r')
        out.write(frame)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()