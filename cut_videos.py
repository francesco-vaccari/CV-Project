import cv2

video = 'videos/out11.mp4'
save_video = 'videos/out11_cut.mp4'

start_time = 30
end_time = 240

cap = cv2.VideoCapture(video)


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4

out = cv2.VideoWriter(save_video, fourcc, fps, (width, height))

start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

current_frame = start_frame
while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    current_frame += 1
    print(f"\rProgress: {current_frame} / {end_frame}", end='')

cap.release()
out.release()

cv2.destroyAllWindows()