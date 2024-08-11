import numpy as np
import pickle
import cv2
from itertools import combinations
import os
from tqdm import tqdm

video = 'out9_combined.mp4'
save_video = 'out9_transformed.mp4'
folder = 'output'
indexes = [1, 2, 3, 4]



def get_triangle_masks(bgs, points, old_points, M, triangles, old_triangles):
    overlaps = []
    bgx = [(idx, bg) for idx, bg in enumerate(bgs)]
    masks = {idx: [] for idx, _ in bgx}
    for (idx1, bg1), (idx2, bg2) in list(combinations(bgx, 2)):
        # check if not intersecting
        image_intersection = cv2.bitwise_and(bg1, bg2)

        image_mask = np.zeros_like(image_intersection)
        image_mask[np.where(image_intersection >= 1)] = 255
        image_mask_inv = np.ones_like(image_mask) * 255
        image_mask_inv[np.where(image_mask >= 1)] = 0

        image_overlap = np.zeros_like(bg1)
        image_overlap = cv2.addWeighted(bg1, 0.5, bg2, 0.5, 0)

        image_common = cv2.bitwise_and(image_overlap, image_mask)

        if np.count_nonzero(image_common) > 15:
            overlaps.append(image_common)

        masks[idx1].append(image_mask_inv)
        masks[idx2].append(image_mask_inv)

    return masks, overlaps

def blend(frame, bgs, points, old_points, M, triangles, old_triangles):
    new_frame = np.zeros_like(frame)
    if len(bgs) > 1:  # if there are more than 1 background
        masks, overlaps = get_triangle_masks(bgs, points, old_points, M, triangles, old_triangles)

        # draw non-overlapping part of the triangles
        for idx, mask in masks.items():
            triangle_crop = bgs[idx]
            for m in mask:
                triangle_crop = cv2.bitwise_and(triangle_crop, m)

            new_frame = cv2.bitwise_or(new_frame, triangle_crop)

        if len(overlaps) == 1:  # if there is only 1 overlap, draw it
            new_frame = cv2.add(new_frame, overlaps[0])
        else:  # else get masks and draw non-overlapping part of the overlaps
            masks, over = get_triangle_masks(overlaps, points, old_points, M, triangles, old_triangles)
            for idx, mask in masks.items():
                triangle_crop = overlaps[idx]
                for m in mask:
                    triangle_crop = cv2.bitwise_and(triangle_crop, m)

                new_frame = cv2.bitwise_or(new_frame, triangle_crop)

            if 0 < len(over) <= 3:  # take the smallest overlap and draw it
                over = min(over, key=lambda x: np.count_nonzero(x))
                new_frame = cv2.add(new_frame, over)
            elif len(over) > 3:  # otherwise blend all overlaps
                background = np.zeros_like(frame)
                for o in over:
                    background = cv2.add(background, o)
                new_frame = cv2.add(new_frame, background)

    else:
        new_frame = bgs[0]

    return new_frame

def apply_warp(frame, points, old_points, M, triangles, old_triangles):
    bg = np.zeros_like(frame)
    bgs = []

    tri1 = np.float32(old_triangles)
    tri2 = np.float32(triangles)

    current_bg = np.zeros_like(frame)

    # Two rectangles are defined, each circumscribing the old and new triangles, respectively
    # Two rectangles are defined, each circumscribing the old and new triangles, respectively
    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)

    # The coordinates of both triangles are expressed relative to the coordinates of their respective
    # rectangles
    tri1Cropped = []
    tri2Cropped = []

    for i in range(0, 3):
        tri1Cropped.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
        tri2Cropped.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))

        warpMat = M

        # Crop input image
        img1Cropped = frame[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]

        # The transformation is then applied to the old rectangle to obtain the second one using the
        # warpAffine function.
        img2Cropped = cv2.warpAffine(
            img1Cropped,
            warpMat,
            (r2[2], r2[3]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # A mask, sized according to the destination rectangle, is generated to retain only the pixels
        # related to the final triangle, setting the rest to zero
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

        img2Cropped = img2Cropped * mask

        # Copy triangular region of the rectangular patch to the output image
        current_bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = bg[
            r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
        ] * ((1.0, 1.0, 1.0) - mask)

        # Subsequently, the new triangle is removed from the image and replaced by its warped version
        current_bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
            bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2Cropped
        )

        bgs.append(current_bg)
    bg = blend(frame, bgs, points, old_points, M, triangles, old_triangles)
    return bg

def get_all(index):
    points_file = f'points_{index}.pkl'
    old_points_file = f'old_points_{index}.pkl'
    trans_matrix_file = f'trans_matrix_{index}.pkl'
    triangle_file = f'triangles_{index}.pkl'
    old_triangle_file = f'old_triangles_{index}.pkl'

    with open(os.path.join(folder, points_file), 'rb') as f:
        points = pickle.load(f)
        points = np.array(points, np.int32)

    with open(os.path.join(folder, old_points_file), 'rb') as f:
        old_points = pickle.load(f)
        old_points = np.array(old_points, np.int32)

    with open(os.path.join(folder, trans_matrix_file), 'rb') as f:
        trans_matrix = pickle.load(f)
        M = np.array(trans_matrix[0], np.float32)

    with open(os.path.join(folder, triangle_file), 'rb') as f:
        triangles = pickle.load(f)
        triangles = np.array(triangles, np.int32)[0]

    with open(os.path.join(folder, old_triangle_file), 'rb') as f:
        old_triangles = pickle.load(f)
        old_triangles = np.array(old_triangles, np.int32)[0]
    
    return points, old_points, M, triangles, old_triangles


cap = cv2.VideoCapture(video)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for i in indexes:
        points, old_points, M, triangles, old_triangles = get_all(i)
        area = apply_warp(frame, points, old_points, M, triangles, old_triangles)
        mask = np.zeros_like(area)
        mask[np.where(area == 0)] = 255
        mask[np.where(area != 0)] = 0
        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, area)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print('Saving video...')
cap = cv2.VideoCapture(video)
writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    for i in indexes:
        points, old_points, M, triangles, old_triangles = get_all(i)
        area = apply_warp(frame, points, old_points, M, triangles, old_triangles)
        mask = np.zeros_like(area)
        mask[np.where(area == 0)] = 255
        mask[np.where(area != 0)] = 0
        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, area)

    writer.write(frame)

cap.release()
writer.release()
cv2.destroyAllWindows()