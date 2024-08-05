import cv2
import numpy as np

left_video_name = 'videos/out9_left_undistorted.mp4'
right_video_name = 'videos/out9_right_undistorted.mp4'

# Global variables
paused = False
rewind = False
transformation1 = {'rotate': 0, 'scale': 1.0, 'tx': 0, 'ty': 0, 'perspective_1': 10000, 'perspective_2': 10000, 'perspective_3': 10000}
transformation2 = {'rotate': 0, 'scale': 1.0, 'tx': 0, 'ty': 0, 'perspective_1': 10000, 'perspective_2': 10000, 'perspective_3': 10000}

def on_change(x):
    pass

def apply_transformations(frame, n):
    # Get the transformation parameters
    if n == 1:
        angle = transformation1['rotate']
        scale = transformation1['scale']
        tx = transformation1['tx']
        ty = transformation1['ty']
        perspective_1 = transformation1['perspective_1'] / 10000 if transformation1['perspective_1'] != 0 else 1
        perspective_2 = transformation1['perspective_2'] / 10000 if transformation1['perspective_2'] != 0 else 1
        perspective_3 = transformation1['perspective_3'] / 10000 if transformation1['perspective_3'] != 0 else 1
        
        # Get the image dimensions
        h, w = frame.shape[:2]
        
        # Rotation matrix
        M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        
        # Translation matrix
        M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply rotation and then translation
        transformed_frame = cv2.warpAffine(frame, M_rotate, (w, h))
        transformed_frame = cv2.warpAffine(transformed_frame, M_translate, (w, h))

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[0, 0], [w/perspective_1, 0], [0, h/perspective_2], [w/perspective_3, h/perspective_3]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(transformed_frame, M, (w, h))
        
        return transformed_frame
    if n == 2:
        angle = transformation2['rotate']
        scale = transformation2['scale']
        tx = transformation2['tx']
        ty = transformation2['ty']
        perspective_1 = transformation2['perspective_1'] / 10000 if transformation2['perspective_1'] != 0 else 1
        perspective_2 = transformation2['perspective_2'] / 10000 if transformation2['perspective_2'] != 0 else 1
        perspective_3 = transformation2['perspective_3'] / 10000 if transformation2['perspective_3'] != 0 else 1
        
        # Get the image dimensions
        h, w = frame.shape[:2]
        
        # Rotation matrix
        M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        
        # Translation matrix
        M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply rotation and then translation
        transformed_frame = cv2.warpAffine(frame, M_rotate, (w, h))
        transformed_frame = cv2.warpAffine(transformed_frame, M_translate, (w, h))

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[0, 0], [w/perspective_1, 0], [0, h/perspective_2], [w/perspective_3, h/perspective_3]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(transformed_frame, M, (w, h))
        
        return transformed_frame

def update_transformation(val):
    transformation1['rotate'] = cv2.getTrackbarPos('1:Rotate', 'Video')
    transformation1['scale'] = cv2.getTrackbarPos('1:Scale', 'Video') / 100.0
    transformation1['tx'] = cv2.getTrackbarPos('1:Tx', 'Video')
    transformation1['ty'] = cv2.getTrackbarPos('1:Ty', 'Video')
    transformation1['perspective_1'] = cv2.getTrackbarPos('1:Persp 1', 'Video')
    transformation1['perspective_2'] = cv2.getTrackbarPos('1:Persp 2', 'Video')
    transformation1['perspective_3'] = cv2.getTrackbarPos('1:Persp 3', 'Video')
    transformation2['rotate'] = cv2.getTrackbarPos('2:Rotate', 'Video')
    transformation2['scale'] = cv2.getTrackbarPos('2:Scale', 'Video') / 100.0
    transformation2['tx'] = cv2.getTrackbarPos('2:Tx', 'Video')
    transformation2['ty'] = cv2.getTrackbarPos('2:Ty', 'Video')
    transformation2['perspective_1'] = cv2.getTrackbarPos('2:Persp 1', 'Video')
    transformation2['perspective_2'] = cv2.getTrackbarPos('2:Persp 2', 'Video')
    transformation2['perspective_3'] = cv2.getTrackbarPos('2:Persp 3', 'Video')

# Load the videos
cap1 = cv2.VideoCapture(left_video_name)
cap2 = cv2.VideoCapture(right_video_name)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one of the videos.")
    exit()

# Create a window
cv2.namedWindow('Video')

# Create trackbars for transformations
cv2.createTrackbar('1:Rotate', 'Video', 0, 360, on_change)
cv2.createTrackbar('1:Scale', 'Video', 100, 200, on_change) # scale factor from 0.0 to 2.0
cv2.createTrackbar('1:Tx', 'Video', 0, 1000, on_change) # translation x
cv2.createTrackbar('1:Ty', 'Video', 0, 1000, on_change) # translation y
cv2.createTrackbar('1:Persp 1', 'Video', 10000, 100000, on_change)
cv2.createTrackbar('1:Persp 2', 'Video', 10000, 100000, on_change)
cv2.createTrackbar('1:Persp 3', 'Video', 10000, 100000, on_change)
cv2.createTrackbar('2:Rotate', 'Video', 0, 360, on_change)
cv2.createTrackbar('2:Scale', 'Video', 100, 200, on_change) # scale factor from 0.0 to 2.0
cv2.createTrackbar('2:Tx', 'Video', 0, 1000, on_change) # translation x
cv2.createTrackbar('2:Ty', 'Video', 0, 1000, on_change) # translation y
cv2.createTrackbar('2:Persp 1', 'Video', 10000, 100000, on_change)
cv2.createTrackbar('2:Persp 2', 'Video', 10000, 100000, on_change)
cv2.createTrackbar('2:Persp 3', 'Video', 10000, 100000, on_change)

while True:
    if not paused:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Reached end of one of the videos, rewinding...")
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
        
        transformed_frame1 = apply_transformations(frame1, 1)
        transformed_frame2 = apply_transformations(frame2, 2)
        
        combined_frame = np.hstack((transformed_frame1, transformed_frame2))
        cv2.imshow('Video', combined_frame)
    
    # Wait for 10 ms
    key = cv2.waitKey(10) & 0xFF
    
    # Check for user input
    if key == ord('q'):
        break
    elif key == ord('p') or key == ord(' '):
        paused = not paused
    elif key == ord('r'):
        rewind = True
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        paused = False
    elif key == ord('a'):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, cap1.get(cv2.CAP_PROP_POS_FRAMES) - 5 * cap1.get(cv2.CAP_PROP_FPS))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, cap2.get(cv2.CAP_PROP_POS_FRAMES) - 5 * cap2.get(cv2.CAP_PROP_FPS))
    elif key == ord('d'):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, cap1.get(cv2.CAP_PROP_POS_FRAMES) + 5 * cap1.get(cv2.CAP_PROP_FPS))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, cap2.get(cv2.CAP_PROP_POS_FRAMES) + 5 * cap2.get(cv2.CAP_PROP_FPS))
    
    # Update transformations based on trackbar positions
    update_transformation(0)

# Release video capture and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()