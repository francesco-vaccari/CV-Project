import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard (number of inside corners per row and column)
CHECKERBOARD = (6, 9)
# Define the real size of the checkerboard squares in some unit (e.g., millimeters)
square_size = 28  # This value should be the real size of the checkerboard squares

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create vectors to store 3D points and 2D points for all images
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

# Prepare grid and points to display
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# Extracting path of individual images stored in a given directory
images = glob.glob('out9safe_frame2/*.jpg')

# Variable to store the shape of the grayscale images
gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Store the shape of the gray image
    if gray_shape is None:
        gray_shape = gray.shape[::-1]

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    # If found, add object points and image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if gray_shape is not None and len(objpoints) > 0 and len(imgpoints) > 0:
    # Perform camera calibration by passing the value of known 3D points (objpoints) and corresponding pixel coordinates of the detected corners (imgpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # Save the camera calibration results for later use
    np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibration successful. Results saved to calibration_data.npz")
else:
    print("Error: No valid checkerboard patterns found.")