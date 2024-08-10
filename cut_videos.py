import cv2

video = 'videos/out9.mp4'
save_video = 'videos/out9_cut.mp4'

video2 = 'videos/out10.mp4'
save_video2 = 'videos/out10_cut.mp4'

video3 = 'videos/out11.mp4'
save_video3 = 'videos/out11_cut.mp4'

cap = cv2.VideoCapture(video)
writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

cap2 = cv2.VideoCapture(video2)
writer2 = cv2.VideoWriter(save_video2, cv2.VideoWriter_fourcc(*'mp4v'), cap2.get(cv2.CAP_PROP_FPS), (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))))

cap3 = cv2.VideoCapture(video3)
writer3 = cv2.VideoWriter(save_video3, cv2.VideoWriter_fourcc(*'mp4v'), cap3.get(cv2.CAP_PROP_FPS), (int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))))


# save only after 5 minutes
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
start_frame = 30 * frame_rate
end_frame = 240 * frame_rate

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= start_frame and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        print(f'Saving video {video} from frame {start_frame} to frame {end_frame}, Progress: {cap.get(cv2.CAP_PROP_POS_FRAMES)}/{end_frame}', end='\r')
        writer.write(frame)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
        break

cap.release()
writer.release()

while cap2.isOpened():
    ret, frame = cap2.read()
    if not ret:
        break

    if cap2.get(cv2.CAP_PROP_POS_FRAMES) >= start_frame and cap2.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        print(f'Saving video {video2} from frame {start_frame} to frame {end_frame}, Progress: {cap2.get(cv2.CAP_PROP_POS_FRAMES)}/{end_frame}', end='\r')
        writer2.write(frame)
    if cap2.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
        break

cap2.release()
writer2.release()

while cap3.isOpened():
    ret, frame = cap3.read()
    if not ret:
        break

    if cap3.get(cv2.CAP_PROP_POS_FRAMES) >= start_frame and cap3.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        print(f'Saving video {video3} from frame {start_frame} to frame {end_frame}, Progress: {cap3.get(cv2.CAP_PROP_POS_FRAMES)}/{end_frame}', end='\r')
        writer3.write(frame)
    if cap3.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
        break

cap3.release()
writer3.release()

cv2.destroyAllWindows()