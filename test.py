import cv2
import numpy as np
import time
import tqdm

cap = cv2.VideoCapture('videos/out9_left.mp4')
calibration_path = 'videos/out9safe_left_frames.npz'
cap2 = cv2.VideoCapture('videos/out9_right.mp4')
calibration_path2 = 'videos/out9safe_right_frames.npz'

saving_path = 'videos/out9_combined.mp4'

precision = 10000



def on_trackbar(val):
    pass

calibration_data = np.load(calibration_path)
k = calibration_data['mtx']
d = calibration_data['dist']
orig_k = k.copy()
orig_d = d.copy()

calibration_data2 = np.load(calibration_path2)
k2 = calibration_data2['mtx']
d2 = calibration_data2['dist']
orig_k2 = k2.copy()
orig_d2 = d2.copy()

tx = 0
ty = 0
tx2 = 0
ty2 = 0

cv2.namedWindow('Trackbars', cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('1:K1', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:K2', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:Cx', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:Cy', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:D1', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:D2', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:D3', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:D4', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:D5', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('1:Tx', 'Trackbars', 0, 1000, on_trackbar)
cv2.createTrackbar('1:Ty', 'Trackbars', 0, 1000, on_trackbar)

cv2.createTrackbar('2:K1', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:K2', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:Cx', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:Cy', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:D1', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:D2', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:D3', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:D4', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:D5', 'Trackbars', 0, 2*precision, on_trackbar)
cv2.createTrackbar('2:Tx', 'Trackbars', 0, 1000, on_trackbar)
cv2.createTrackbar('2:Ty', 'Trackbars', 0, 1000, on_trackbar)

cv2.setTrackbarPos('1:K1', 'Trackbars', int(orig_k[0, 0]) + precision)
cv2.setTrackbarPos('1:K2', 'Trackbars', int(orig_k[1, 1]) + precision)
cv2.setTrackbarPos('1:Cx', 'Trackbars', int(orig_k[0, 2]) + precision)
cv2.setTrackbarPos('1:Cy', 'Trackbars', int(orig_k[1, 2]) + precision)
cv2.setTrackbarPos('1:D1', 'Trackbars', int(precision + orig_d[0, 0]*precision))
cv2.setTrackbarPos('1:D2', 'Trackbars', int(precision + orig_d[0, 1]*precision))
cv2.setTrackbarPos('1:D3', 'Trackbars', int(precision + orig_d[0, 2]*precision))
cv2.setTrackbarPos('1:D4', 'Trackbars', int(precision + orig_d[0, 3]*precision))
cv2.setTrackbarPos('1:D5', 'Trackbars', int(precision + orig_d[0, 4]*precision))
cv2.setTrackbarPos('1:Tx', 'Trackbars', tx)
cv2.setTrackbarPos('1:Ty', 'Trackbars', ty)

cv2.setTrackbarPos('2:K1', 'Trackbars', int(orig_k2[0, 0]) + precision)
cv2.setTrackbarPos('2:K2', 'Trackbars', int(orig_k2[1, 1]) + precision)
cv2.setTrackbarPos('2:Cx', 'Trackbars', int(orig_k2[0, 2]) + precision)
cv2.setTrackbarPos('2:Cy', 'Trackbars', int(orig_k2[1, 2]) + precision)
cv2.setTrackbarPos('2:D1', 'Trackbars', int(precision + orig_d2[0, 0]*precision))
cv2.setTrackbarPos('2:D2', 'Trackbars', int(precision + orig_d2[0, 1]*precision))
cv2.setTrackbarPos('2:D3', 'Trackbars', int(precision + orig_d2[0, 2]*precision))
cv2.setTrackbarPos('2:D4', 'Trackbars', int(precision + orig_d2[0, 3]*precision))
cv2.setTrackbarPos('2:D5', 'Trackbars', int(precision + orig_d2[0, 4]*precision))
cv2.setTrackbarPos('2:Tx', 'Trackbars', tx2)
cv2.setTrackbarPos('2:Ty', 'Trackbars', ty2)


def get_params():
    global k, d, k2, d2, tx, ty, tx2, ty2
    k[0, 0] = cv2.getTrackbarPos('1:K1', 'Trackbars') - precision
    k[1, 1] = cv2.getTrackbarPos('1:K2', 'Trackbars') - precision
    k[0, 2] = cv2.getTrackbarPos('1:Cx', 'Trackbars') - precision
    k[1, 2] = cv2.getTrackbarPos('1:Cy', 'Trackbars') - precision
    d[0, 0] = (cv2.getTrackbarPos('1:D1', 'Trackbars') - precision) / float(precision)
    d[0, 1] = (cv2.getTrackbarPos('1:D2', 'Trackbars') - precision) / float(precision)
    d[0, 2] = (cv2.getTrackbarPos('1:D3', 'Trackbars') - precision) / float(precision)
    d[0, 3] = (cv2.getTrackbarPos('1:D4', 'Trackbars') - precision) / float(precision)
    d[0, 4] = (cv2.getTrackbarPos('1:D5', 'Trackbars') - precision) / float(precision)
    tx = cv2.getTrackbarPos('1:Tx', 'Trackbars')
    ty = cv2.getTrackbarPos('1:Ty', 'Trackbars')
    k2[0, 0] = cv2.getTrackbarPos('2:K1', 'Trackbars') - precision
    k2[1, 1] = cv2.getTrackbarPos('2:K2', 'Trackbars') - precision
    k2[0, 2] = cv2.getTrackbarPos('2:Cx', 'Trackbars') - precision
    k2[1, 2] = cv2.getTrackbarPos('2:Cy', 'Trackbars') - precision
    d2[0, 0] = (cv2.getTrackbarPos('2:D1', 'Trackbars') - precision) / float(precision)
    d2[0, 1] = (cv2.getTrackbarPos('2:D2', 'Trackbars') - precision) / float(precision)
    d2[0, 2] = (cv2.getTrackbarPos('2:D3', 'Trackbars') - precision) / float(precision)
    d2[0, 3] = (cv2.getTrackbarPos('2:D4', 'Trackbars') - precision) / float(precision)
    d2[0, 4] = (cv2.getTrackbarPos('2:D5', 'Trackbars') - precision) / float(precision)
    tx2 = cv2.getTrackbarPos('2:Tx', 'Trackbars')
    ty2 = cv2.getTrackbarPos('2:Ty', 'Trackbars')

def transform(frame, k, d, tx, ty):
        # frame = cv2.undistort(frame, k, d, None)

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
    
    frame = transform(frame, k, d, tx, ty)
    frame2 = transform(frame2, k2, d2, tx2, ty2)
    
    # cv2.imshow('Image', frame)
    # cv2.imshow('Image2', frame2)
    combined_frame = np.hstack((frame, frame2))
    cv2.imshow('Image', combined_frame)
    
    # Wait for 1 ms and check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        save = True

# Release the video capture and close all windows
cap.release()
cap2.release()
cv2.destroyAllWindows()

if save:
    cap = cv2.VideoCapture('videos/out9_left.mp4')
    cap2 = cv2.VideoCapture('videos/out9_right.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(saving_path, fourcc, 24, (int(cap.get(3)), int(cap.get(4))))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm.tqdm(total=total_frames, desc='Saving Video', unit='frame')
    
    while cap.isOpened():
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        if not ret or not ret2:
            break

        get_params()

        frame = transform(frame, k, d, tx, ty)
        frame2 = transform(frame2, k2, d2, tx2, ty2)

        combined_frame = np.hstack((frame, frame2))
        out.write(combined_frame)
        
        progress_bar.update(1)

    cap.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    print('Video saved to ' + saving_path)