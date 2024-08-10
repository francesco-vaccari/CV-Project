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


frame_rate = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(30 * frame_rate)
end_frame = int(240 * frame_rate)

frame_rate2 = cap2.get(cv2.CAP_PROP_FPS)
start_frame2 = int(30 * frame_rate2)
end_frame2 = int(240 * frame_rate2)

frame_rate3 = cap3.get(cv2.CAP_PROP_FPS)
start_frame3 = int(30 * frame_rate3)
end_frame3 = int(240 * frame_rate3)

print(f'Saving video {video} from frame {start_frame} to frame {end_frame}')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if cap.get(cv2.CAP_PROP_POS_FRAMES) < start_frame:
        print(f'Skipping frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}', end='\r')
        continue

    writer.write(frame)
    print(f'Writing frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}', end='\r')

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
        break

cap.release()
writer.release()

print(f'Saving video {video2} from frame {start_frame2} to frame {end_frame2}')
while cap2.isOpened():
    ret, frame = cap2.read()
    if not ret:
        break

    if cap2.get(cv2.CAP_PROP_POS_FRAMES) < start_frame2:
        print(f'Skipping frame {cap2.get(cv2.CAP_PROP_POS_FRAMES)}', end='\r')
        continue

    writer2.write(frame)
    print(f'Writing frame {cap2.get(cv2.CAP_PROP_POS_FRAMES)}', end='\r')

    if cap2.get(cv2.CAP_PROP_POS_FRAMES) == end_frame2:
        break

cap2.release()
writer2.release()

print(f'Saving video {video3} from frame {start_frame3} to frame {end_frame3}')
while cap3.isOpened():
    ret, frame = cap3.read()
    if not ret:
        break
    
    if cap3.get(cv2.CAP_PROP_POS_FRAMES) < start_frame3:
        print(f'Skipping frame {cap3.get(cv2.CAP_PROP_POS_FRAMES)}', end='\r')
        continue

    writer3.write(frame)
    print(f'Writing frame {cap3.get(cv2.CAP_PROP_POS_FRAMES)}', end='\r')

    if cap3.get(cv2.CAP_PROP_POS_FRAMES) == end_frame3:
        break

cap3.release()
writer3.release()

cv2.destroyAllWindows()