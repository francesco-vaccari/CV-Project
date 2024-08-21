import cv2

video = 'videos/refined2_short.mp4'
save_file = f'annotations_ball.txt'

cap = cv2.VideoCapture(video)
center_pos = (0, 0)
radius = 0
next_frame = True
button_down = False
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = None

file = open(save_file, 'a')

def mouse_callback(event, x, y, flags, param):
    global center_pos, radius, button_down
    if event == cv2.EVENT_LBUTTONDOWN:
        center_pos = (x, y)
        button_down = True
    if event == cv2.EVENT_MOUSEMOVE:
        if button_down:
            radius = int(((x - center_pos[0])**2 + (y - center_pos[1])**2)**0.5)
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
    cv2.circle(frame_copy, center_pos, radius, (0, 255, 0), 2)
    cv2.imshow('frame', frame_copy)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print(f'Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}\t Center: {center_pos}\t Radius: {radius}')
        next_frame = True
        button_down = False
        file.write(f'{int(cap.get(cv2.CAP_PROP_POS_FRAMES))},{center_pos[0]},{center_pos[1]},{radius}\n')
    if key == ord('a'):
        center_pos = (center_pos[0] - 1, center_pos[1])
    if key == ord('d'):
        center_pos = (center_pos[0] + 1, center_pos[1])
    if key == ord('w'):
        center_pos = (center_pos[0], center_pos[1] - 1)
    if key == ord('s'):
        center_pos = (center_pos[0], center_pos[1] + 1)
    if key == ord('j'):
        radius = max(0, radius - 1)
    if key == ord('l'):
        radius += 1
    if key == ord('z'):
        center_pos = (0, 0)
        radius = 0

file.close()
cap.release()
cv2.destroyAllWindows()
