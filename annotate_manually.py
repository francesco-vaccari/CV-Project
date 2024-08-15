import cv2 

video = 'videos/refined2_short.mp4'
team = 1
player = 1
save_file = f'annotations_{team}_{player}.txt'



cap = cv2.VideoCapture(video)
start_pos = (0, 0)
end_pos = (0, 0)
next_frame = True
button_down = False
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = None

file = open(save_file, 'a')

def mouse_callback(event, x, y, flags, param):
    global start_pos, end_pos, button_down
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pos = (x, y)
        button_down = True
    if event == cv2.EVENT_MOUSEMOVE:
        if button_down:
            end_pos = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        button_down = False

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', mouse_callback)

while cap.isOpened():
    if next_frame:
        next_frame = False
        ret, frame = cap.read()
        if not ret:
            break
    
    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, start_pos, end_pos, (0, 255, 0), 4)
    cv2.imshow('frame', frame_copy)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print(f'Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}\t Start: {start_pos}\t End: {end_pos}')
        next_frame = True
        button_down = False
        file.write(f'{int(cap.get(cv2.CAP_PROP_POS_FRAMES))},{start_pos[0]},{start_pos[1]},{end_pos[0]},{end_pos[1]}\n')
    if key == ord('a'):
        # move left box by 1 pixel
        start_pos = (start_pos[0] - 1, start_pos[1])
        end_pos = (end_pos[0] - 1, end_pos[1])
    if key == ord('d'):
        # move right box by 1 pixel
        start_pos = (start_pos[0] + 1, start_pos[1])
        end_pos = (end_pos[0] + 1, end_pos[1])
    if key == ord('w'):
        # move up box by 1 pixel
        start_pos = (start_pos[0], start_pos[1] - 1)
        end_pos = (end_pos[0], end_pos[1] - 1)
    if key == ord('s'):
        # move down box by 1 pixel
        start_pos = (start_pos[0], start_pos[1] + 1)
        end_pos = (end_pos[0], end_pos[1] + 1)
    if key == ord('j'):
        # decrease left edge by 1 pixel
        start_pos = (start_pos[0] - 1, start_pos[1])
    if key == ord('l'):
        # decrease right edge by 1 pixel
        end_pos = (end_pos[0] + 1, end_pos[1])
    if key == ord('i'):
        # decrease top edge by 1 pixel
        start_pos = (start_pos[0], start_pos[1] - 1)
    if key == ord('k'):
        # decrease bottom edge by 1 pixel
        end_pos = (end_pos[0], end_pos[1] + 1)
    if key == ord('f'):
        # increase left edge by 1 pixel
        start_pos = (start_pos[0] + 1, start_pos[1])
    if key == ord('h'):
        # increase right edge by 1 pixel
        end_pos = (end_pos[0] - 1, end_pos[1])
    if key == ord('t'):
        # increase top edge by 1 pixel
        start_pos = (start_pos[0], start_pos[1] + 1)
    if key == ord('g'):
        # increase bottom edge by 1 pixel
        end_pos = (end_pos[0], end_pos[1] - 1)

file.close()