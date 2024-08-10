import cv2
import numpy as np
import tqdm

video_left = 'videos/out9_cut_left.mp4'
video_right = 'videos/out9_cut_right.mp4'
saving_path = 'videos/out9_cut_combined.mp4'



cap = cv2.VideoCapture(video_left)
cap2 = cv2.VideoCapture(video_right)


def on_trackbar(val):
    pass

tx = 0
ty = 0
tx2 = 0
ty2 = 0

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

cv2.createTrackbar('1:Tx', 'Image', 0, 1000, on_trackbar)
cv2.createTrackbar('1:Ty', 'Image', 0, 1000, on_trackbar)
cv2.createTrackbar('2:Tx', 'Image', 0, 1000, on_trackbar)
cv2.createTrackbar('2:Ty', 'Image', 0, 1000, on_trackbar)

cv2.setTrackbarPos('1:Tx', 'Image', tx)
cv2.setTrackbarPos('1:Ty', 'Image', ty)
cv2.setTrackbarPos('2:Tx', 'Image', tx2)
cv2.setTrackbarPos('2:Ty', 'Image', ty2)

def get_params():
    global tx, ty, tx2, ty2
    tx = cv2.getTrackbarPos('1:Tx', 'Image')
    ty = cv2.getTrackbarPos('1:Ty', 'Image')
    tx2 = cv2.getTrackbarPos('2:Tx', 'Image')
    ty2 = cv2.getTrackbarPos('2:Ty', 'Image')

def transform(frame, tx, ty):
    h, w = frame.shape[:2]
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    frame = cv2.warpAffine(frame, M_translate, (w, h))
    return frame

save = False

while cap.isOpened():
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if not ret or not ret2:
        break

    get_params()
    
    frame = transform(frame, tx, ty)
    frame2 = transform(frame2, tx2, ty2)
    
    combined_frame = np.hstack((frame, frame2))
    cv2.imshow('Image', combined_frame)
    
    # line = cv2.line(combined_frame, (0, int(combined_frame.shape[0]/2)), (combined_frame.shape[1], int(combined_frame.shape[0]/2)), (0, 255, 0), 2)
    # cv2.imshow('Image', line)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 5 * cap.get(cv2.CAP_PROP_FPS))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, cap2.get(cv2.CAP_PROP_POS_FRAMES) - 5 * cap2.get(cv2.CAP_PROP_FPS))
    elif key == ord('d'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 5 * cap.get(cv2.CAP_PROP_FPS))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, cap2.get(cv2.CAP_PROP_POS_FRAMES) + 5 * cap2.get(cv2.CAP_PROP_FPS))
    if key == ord('q'):
        break
    if key == ord(' '):
        save = True
        break

tx = cv2.getTrackbarPos('1:Tx', 'Image')
ty = cv2.getTrackbarPos('1:Ty', 'Image')
tx2 = cv2.getTrackbarPos('2:Tx', 'Image')
ty2 = cv2.getTrackbarPos('2:Ty', 'Image')

cap.release()
cap2.release()
cv2.destroyAllWindows()

if save:
    print('Saving video...')

    # Reopen the video files for processing
    cap = cv2.VideoCapture(video_left)
    cap2 = cv2.VideoCapture(video_right)

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    out = cv2.VideoWriter(saving_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width + width2, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm.tqdm(total=total_frames, desc='Saving video')

    while cap.isOpened():
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        if not ret or not ret2:
            break
        
        frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        progress = int((frame_count / total_frames) * 100)
        
        frame = transform(frame, tx, ty)
        frame2 = transform(frame2, tx2, ty2)
        
        combined_frame = np.hstack((frame, frame2))

        out.write(combined_frame)
        
        progress_bar.update(1)

    cap.release()
    cap2.release()
    out.release()

cv2.destroyAllWindows()