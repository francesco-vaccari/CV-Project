import cv2
import tqdm

video = 'videos/combined.mp4'
save_video = 'videos/refined.mp4'

cap = cv2.VideoCapture(video)

top_line = 0
bottom_line = 0
left_line = 0
right_line = 0
mult = 1

bottom_line = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
right_line = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    try:
        frame = frame[top_line:bottom_line, left_line:right_line]
        cv2.imshow('frame', frame)
    except:
        print('Out of bounds')

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break
    if key == ord('w'):
        top_line -= 1 * mult
    if key == ord('s'):
        top_line += 1 * mult
    if key == ord('a'):
        left_line -= 1 * mult
    if key == ord('d'):
        left_line += 1 * mult
    if key == ord('i'):
        bottom_line -= 1 * mult
    if key == ord('k'):
        bottom_line += 1 * mult
    if key == ord('j'):
        right_line -= 1 * mult
    if key == ord('l'):
        right_line += 1 * mult
    if key == ord('+'):
        mult *= 2
    if key == ord('-'):
        mult = mult // 2
        if mult < 1:
            mult = 1

cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(video)
writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (right_line - left_line, bottom_line - top_line))

progress = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[top_line:bottom_line, left_line:right_line]

    writer.write(frame)
    progress.update(1)

cap.release()
writer.release()
cv2.destroyAllWindows()