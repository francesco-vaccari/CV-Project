import cv2
import numpy as np

top = "videos/out9_combined.mp4"
center = "videos/out10_combined.mp4"
bottom = "videos/out11_combined.mp4"




top_video = cv2.VideoCapture(top)
center_video = cv2.VideoCapture(center)
bottom_video = cv2.VideoCapture(bottom)

params = {'r1': 0, 'r2': 0, 'r3': 0, 's1': 100, 's2': 100, 's3': 100, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'x3': 0, 'y3': 0}
mult = 1

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# cv2.namedWindow('TrackBars', cv2.WINDOW_NORMAL)

# cv2.createTrackbar('Top rotation', 'TrackBars', 0, 360, lambda x: None)
# cv2.createTrackbar('Center rotation', 'TrackBars', 0, 360, lambda x: None)
# cv2.createTrackbar('Bottom rotation', 'TrackBars', 0, 360, lambda x: None)
# cv2.createTrackbar('Top scale', 'TrackBars', 100, 200, lambda x: None)
# cv2.createTrackbar('Center scale', 'TrackBars', 100, 200, lambda x: None)
# cv2.createTrackbar('Bottom scale', 'TrackBars', 100, 200, lambda x: None)
# cv2.createTrackbar('Top x', 'TrackBars', 0, 1000, lambda x: None)
# cv2.createTrackbar('Top y', 'TrackBars', 0, 1000, lambda x: None)
# cv2.createTrackbar('Center x', 'TrackBars', 0, 1000, lambda x: None)
# cv2.createTrackbar('Center y', 'TrackBars', 0, 1000, lambda x: None)
# cv2.createTrackbar('Bottom x', 'TrackBars', 0, 1000, lambda x: None)
# cv2.createTrackbar('Bottom y', 'TrackBars', 0, 1000, lambda x: None)

# def update_params():
#     params['r1'] = cv2.getTrackbarPos('Top rotation', 'TrackBars')
#     params['r2'] = cv2.getTrackbarPos('Center rotation', 'TrackBars')
#     params['r3'] = cv2.getTrackbarPos('Bottom rotation', 'TrackBars')
#     params['s1'] = cv2.getTrackbarPos('Top scale', 'TrackBars')
#     params['s2'] = cv2.getTrackbarPos('Center scale', 'TrackBars')
#     params['s3'] = cv2.getTrackbarPos('Bottom scale', 'TrackBars')
#     params['x1'] = cv2.getTrackbarPos('Top x', 'TrackBars')
#     params['y1'] = cv2.getTrackbarPos('Top y', 'TrackBars')
#     params['x2'] = cv2.getTrackbarPos('Center x', 'TrackBars')
#     params['y2'] = cv2.getTrackbarPos('Center y', 'TrackBars')
#     params['x3'] = cv2.getTrackbarPos('Bottom x', 'TrackBars')
#     params['y3'] = cv2.getTrackbarPos('Bottom y', 'TrackBars')


def apply_transformation(top, center, bottom):

    top = cv2.resize(top, (top.shape[1] * params['s1'] // 100, top.shape[0] * params['s1'] // 100))
    top = cv2.warpAffine(top, cv2.getRotationMatrix2D((top.shape[1] // 2, top.shape[0] // 2), params['r1'], 1), (top.shape[1], top.shape[0]))
    top = cv2.warpAffine(top, np.float32([[1, 0, params['x1']], [0, 1, params['y1']]]), (top.shape[1], top.shape[0]))

    center = cv2.resize(center, (center.shape[1] * params['s2'] // 100, center.shape[0] * params['s2'] // 100))
    center = cv2.warpAffine(center, cv2.getRotationMatrix2D((center.shape[1] // 2, center.shape[0] // 2), params['r2'], 1), (center.shape[1], center.shape[0]))
    center = cv2.warpAffine(center, np.float32([[1, 0, params['x2']], [0, 1, params['y2']]]), (center.shape[1], center.shape[0]))

    bottom = cv2.resize(bottom, (bottom.shape[1] * params['s3'] // 100, bottom.shape[0] * params['s3'] // 100))
    bottom = cv2.warpAffine(bottom, cv2.getRotationMatrix2D((bottom.shape[1] // 2, bottom.shape[0] // 2), params['r3'], 1), (bottom.shape[1], bottom.shape[0]))
    bottom = cv2.warpAffine(bottom, np.float32([[1, 0, params['x3']], [0, 1, params['y3']]]), (bottom.shape[1], bottom.shape[0]))

    return top, center, bottom


while True:
    ret1, top = top_video.read()
    ret2, center = center_video.read()
    ret3, bottom = bottom_video.read()

    if not ret1 or not ret2 or not ret3:
        break
        
    top = cv2.rotate(top, cv2.ROTATE_180)

    top, center, bottom = apply_transformation(top, center, bottom)

    # pad frames to make them have same width
    max_width = max(top.shape[1], center.shape[1], bottom.shape[1])
    top = cv2.copyMakeBorder(top, 0, 0, 0, max_width - top.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    center = cv2.copyMakeBorder(center, 0, 0, 0, max_width - center.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    bottom = cv2.copyMakeBorder(bottom, 0, 0, 0, max_width - bottom.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))

    stitched_frame = np.vstack((top, center, bottom))

    cv2.imshow("Image", stitched_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        mult *= -1
    if key == ord('q'):
        params['r1'] += 1 * mult
    if key == ord('w'):
        params['r2'] += 1 * mult
    if key == ord('e'):
        params['r3'] += 1 * mult
    if key == ord('a'):
        params['s1'] += 1 * mult
    if key == ord('s'):
        params['s2'] += 1 * mult
    if key == ord('d'):
        params['s3'] += 1 * mult
    if key == ord('z'):
        params['x1'] += 1 * mult
    if key == ord('x'):
        params['y1'] += 1 * mult
    if key == ord('c'):
        params['x2'] += 1 * mult
    if key == ord('v'):
        params['y2'] += 1 * mult
    if key == ord('b'):
        params['x3'] += 1 * mult
    if key == ord('n'):
        params['y3'] += 1 * mult