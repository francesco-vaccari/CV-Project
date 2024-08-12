import cv2
import numpy as np
import tqdm

top = "videos/out9_transformed.mp4"
center = "videos/out10_transformed_fps_adjusted.mp4"
bottom = "videos/out11_transformed.mp4"

save_video = "videos/combined.mp4"

pad = 700



top_video = cv2.VideoCapture(top)
center_video = cv2.VideoCapture(center)
bottom_video = cv2.VideoCapture(bottom)

params1 = {'r': 0, 's': 100, 'x': 0, 'y': 0}
params2 = {'r': 0, 's': 100, 'x': 0, 'y': 0}
params3 = {'r': 0, 's': 100, 'x': 0, 'y': 0}
params = [params1, params2, params3]
focus = 0
order = 1

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

def draw_on_canvas(canvas, start_x, start_y, end_x, end_y, frame):
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            if np.all(canvas[i][j] == 0):
                canvas[i][j] = frame[i - start_y][j - start_x]
    return canvas

def copy_frames2(canvas, top, top_start, center, center_start, bottom, bottom_start):
    if order == 1 or order == 2:
        canvas = draw_on_canvas(canvas, top_start[0], top_start[1], top_start[0] + top.shape[1], top_start[1] + top.shape[0], top)
        if order == 1:
            canvas = draw_on_canvas(canvas, center_start[0], center_start[1], center_start[0] + center.shape[1], center_start[1] + center.shape[0], center)
            canvas = draw_on_canvas(canvas, bottom_start[0], bottom_start[1], bottom_start[0] + bottom.shape[1], bottom_start[1] + bottom.shape[0], bottom)
        if order == 2:
            canvas = draw_on_canvas(canvas, bottom_start[0], bottom_start[1], bottom_start[0] + bottom.shape[1], bottom_start[1] + bottom.shape[0], bottom)
            canvas = draw_on_canvas(canvas, center_start[0], center_start[1], center_start[0] + center.shape[1], center_start[1] + center.shape[0], center)
    if order == 3 or order == 4:
        canvas = draw_on_canvas(canvas, center_start[0], center_start[1], center_start[0] + center.shape[1], center_start[1] + center.shape[0], center)
        if order == 3:
            canvas = draw_on_canvas(canvas, top_start[0], top_start[1], top_start[0] + top.shape[1], top_start[1] + top.shape[0], top)
            canvas = draw_on_canvas(canvas, bottom_start[0], bottom_start[1], bottom_start[0] + bottom.shape[1], bottom_start[1] + bottom.shape[0], bottom)
        if order == 4:
            canvas = draw_on_canvas(canvas, bottom_start[0], bottom_start[1], bottom_start[0] + bottom.shape[1], bottom_start[1] + bottom.shape[0], bottom)
            canvas = draw_on_canvas(canvas, top_start[0], top_start[1], top_start[0] + top.shape[1], top_start[1] + top.shape[0], top)
    if order == 5 or order == 6:
        canvas = draw_on_canvas(canvas, bottom_start[0], bottom_start[1], bottom_start[0] + bottom.shape[1], bottom_start[1] + bottom.shape[0], bottom)
        if order == 5:
            canvas = draw_on_canvas(canvas, top_start[0], top_start[1], top_start[0] + top.shape[1], top_start[1] + top.shape[0], top)
            canvas = draw_on_canvas(canvas, center_start[0], center_start[1], center_start[0] + center.shape[1], center_start[1] + center.shape[0], center)
        if order == 6:
            canvas = draw_on_canvas(canvas, center_start[0], center_start[1], center_start[0] + center.shape[1], center_start[1] + center.shape[0], center)
            canvas = draw_on_canvas(canvas, top_start[0], top_start[1], top_start[0] + top.shape[1], top_start[1] + top.shape[0], top)

    return canvas

def copy_frames(canvas, top, top_start, center, center_start, bottom, bottom_start):
    if order == 1 or order == 2:
        canvas[top_start[1]:top_start[1] + top.shape[0], top_start[0]:top_start[0] + top.shape[1]] = top
        if order == 1:
            canvas[center_start[1]:center_start[1] + center.shape[0], center_start[0]:center_start[0] + center.shape[1]] = center
            canvas[bottom_start[1]:bottom_start[1] + bottom.shape[0], bottom_start[0]:bottom_start[0] + bottom.shape[1]] = bottom
        if order == 2:
            canvas[bottom_start[1]:bottom_start[1] + bottom.shape[0], bottom_start[0]:bottom_start[0] + bottom.shape[1]] = bottom
            canvas[center_start[1]:center_start[1] + center.shape[0], center_start[0]:center_start[0] + center.shape[1]] = center
    if order == 3 or order == 4:
        canvas[center_start[1]:center_start[1] + center.shape[0], center_start[0]:center_start[0] + center.shape[1]] = center
        if order == 3:
            canvas[top_start[1]:top_start[1] + top.shape[0], top_start[0]:top_start[0] + top.shape[1]] = top
            canvas[bottom_start[1]:bottom_start[1] + bottom.shape[0], bottom_start[0]:bottom_start[0] + bottom.shape[1]] = bottom
        if order == 4:
            canvas[bottom_start[1]:bottom_start[1] + bottom.shape[0], bottom_start[0]:bottom_start[0] + bottom.shape[1]] = bottom
            canvas[top_start[1]:top_start[1] + top.shape[0], top_start[0]:top_start[0] + top.shape[1]] = top
    if order == 5 or order == 6:
        canvas[bottom_start[1]:bottom_start[1] + bottom.shape[0], bottom_start[0]:bottom_start[0] + bottom.shape[1]] = bottom
        if order == 5:
            canvas[top_start[1]:top_start[1] + top.shape[0], top_start[0]:top_start[0] + top.shape[1]] = top
            canvas[center_start[1]:center_start[1] + center.shape[0], center_start[0]:center_start[0] + center.shape[1]] = center
        if order == 6:
            canvas[center_start[1]:center_start[1] + center.shape[0], center_start[0]:center_start[0] + center.shape[1]] = center
            canvas[top_start[1]:top_start[1] + top.shape[0], top_start[0]:top_start[0] + top.shape[1]] = top

    return canvas

while top_video.isOpened():
    ret1, top = top_video.read()
    ret2, center = center_video.read()
    ret3, bottom = bottom_video.read()

    if not ret1 or not ret2 or not ret3:
        break

    top = cv2.rotate(top, cv2.ROTATE_180)

    canvas = np.zeros((top.shape[0] + center.shape[0] + bottom.shape[0] + pad, top.shape[1] + pad, 3), dtype=np.uint8)

    top_start = (0 + pad // 2, 0 + pad // 2)
    center_start = (0 + pad // 2, top.shape[0] + pad // 2)
    bottom_start = (0 + pad // 2, top.shape[0] + center.shape[0] + pad // 2)
    
    top_start = (top_start[0] + params[0]['x'], top_start[1] + params[0]['y'])
    center_start = (center_start[0] + params[1]['x'], center_start[1] + params[1]['y'])
    bottom_start = (bottom_start[0] + params[2]['x'], bottom_start[1] + params[2]['y'])

    top_height = top.shape[0]
    top_width = top.shape[1]
    top = cv2.resize(top, (int(top.shape[1] * params[0]['s'] / 100), int(top.shape[0] * params[0]['s'] / 100)))
    change_in_height = top_height - top.shape[0]
    change_in_width = top_width - top.shape[1]
    top_start = (top_start[0] + change_in_width // 2, top_start[1] + change_in_height // 2)

    center_height = center.shape[0]
    center_width = center.shape[1]
    center = cv2.resize(center, (int(center.shape[1] * params[1]['s'] / 100), int(center.shape[0] * params[1]['s'] / 100)))
    change_in_height = center_height - center.shape[0]
    change_in_width = center_width - center.shape[1]
    center_start = (center_start[0] + change_in_width // 2, center_start[1] + change_in_height // 2)

    bottom_height = bottom.shape[0]
    bottom_width = bottom.shape[1]
    bottom = cv2.resize(bottom, (int(bottom.shape[1] * params[2]['s'] / 100), int(bottom.shape[0] * params[2]['s'] / 100)))
    change_in_height = bottom_height - bottom.shape[0]
    change_in_width = bottom_width - bottom.shape[1]
    bottom_start = (bottom_start[0] + change_in_width // 2, bottom_start[1] + change_in_height // 2)

    top = cv2.warpAffine(top, cv2.getRotationMatrix2D((top.shape[1] / 2, top.shape[0] / 2), params[0]['r'], 1), (top.shape[1], top.shape[0]))
    center = cv2.warpAffine(center, cv2.getRotationMatrix2D((center.shape[1] / 2, center.shape[0] / 2), params[1]['r'], 1), (center.shape[1], center.shape[0]))
    bottom = cv2.warpAffine(bottom, cv2.getRotationMatrix2D((bottom.shape[1] / 2, bottom.shape[0] / 2), params[2]['r'], 1), (bottom.shape[1], bottom.shape[0]))

    canvas = copy_frames(canvas, top, top_start, center, center_start, bottom, bottom_start)
    # canvas = copy_frames2(canvas, top, top_start, center, center_start, bottom, bottom_start)

    cv2.imshow('Image', canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        # translate up
        params[focus]['y'] -= 1
    if key == ord('a'):
        # translate left
        params[focus]['x'] -= 1
    if key == ord('s'):
        # translate down
        params[focus]['y'] += 1
    if key == ord('d'):
        # translate right
        params[focus]['x'] += 1
    if key == ord('q'):
        # rotate leftw
        params[focus]['r'] += 1
    if key == ord('e'):
        # rotate right
        params[focus]['r'] -= 1
    if key == ord('z'):
        # scale up
        params[focus]['s'] += 0.2
    if key == ord('x'):
        # scale down
        params[focus]['s'] -= 0.2
    if key == ord('l'):
        # switch focus to next video
        focus = (focus + 1) % 3
    if key == ord('1'):
        print('Order of drawing: top, center, bottom')
        order = 1
    if key == ord('2'):
        print('Order of drawing: top, bottom, center')
        order = 2
    if key == ord('3'):
        print('Order of drawing: center, top, bottom')
        order = 3
    if key == ord('4'):
        print('Order of drawing: center, bottom, top')
        order = 4
    if key == ord('5'):
        print('Order of drawing: bottom, top, center')
        order = 5
    if key == ord('6'):
        print('Order of drawing: bottom, center, top')
        order = 6
    
    if key == ord(' '):
        break


top_video.release()
center_video.release()
bottom_video.release()

cv2.destroyAllWindows()



top_video = cv2.VideoCapture(top)
center_video = cv2.VideoCapture(center)
bottom_video = cv2.VideoCapture(bottom)
print('Framerate: ', top_video.get(cv2.CAP_PROP_FPS))
print('N frames top center bottom: ', top_video.get(cv2.CAP_PROP_FRAME_COUNT), center_video.get(cv2.CAP_PROP_FRAME_COUNT), bottom_video.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), top_video.get(cv2.CAP_PROP_FPS), (canvas.shape[1], canvas.shape[0]))

progress_bar = tqdm.tqdm(total=int(top_video.get(cv2.CAP_PROP_FRAME_COUNT)))

while top_video.isOpened():
    top_ret, top = top_video.read()
    center_ret, center = center_video.read()
    bottom_ret, bottom = bottom_video.read()

    if not top_ret or not center_ret or not bottom_ret:
        break

    top = cv2.rotate(top, cv2.ROTATE_180)

    canvas = np.zeros((top.shape[0] + center.shape[0] + bottom.shape[0] + pad, top.shape[1] + pad, 3), dtype=np.uint8)

    top_start = (0 + pad // 2, 0 + pad // 2)
    center_start = (0 + pad // 2, top.shape[0] + pad // 2)
    bottom_start = (0 + pad // 2, top.shape[0] + center.shape[0] + pad // 2)
    
    top_start = (top_start[0] + params[0]['x'], top_start[1] + params[0]['y'])
    center_start = (center_start[0] + params[1]['x'], center_start[1] + params[1]['y'])
    bottom_start = (bottom_start[0] + params[2]['x'], bottom_start[1] + params[2]['y'])

    top_height = top.shape[0]
    top_width = top.shape[1]
    top = cv2.resize(top, (int(top.shape[1] * params[0]['s'] / 100), int(top.shape[0] * params[0]['s'] / 100)))
    change_in_height = top_height - top.shape[0]
    change_in_width = top_width - top.shape[1]
    top_start = (top_start[0] + change_in_width // 2, top_start[1] + change_in_height // 2)

    center_height = center.shape[0]
    center_width = center.shape[1]
    center = cv2.resize(center, (int(center.shape[1] * params[1]['s'] / 100), int(center.shape[0] * params[1]['s'] / 100)))
    change_in_height = center_height - center.shape[0]
    change_in_width = center_width - center.shape[1]
    center_start = (center_start[0] + change_in_width // 2, center_start[1] + change_in_height // 2)

    bottom_height = bottom.shape[0]
    bottom_width = bottom.shape[1]
    bottom = cv2.resize(bottom, (int(bottom.shape[1] * params[2]['s'] / 100), int(bottom.shape[0] * params[2]['s'] / 100)))
    change_in_height = bottom_height - bottom.shape[0]
    change_in_width = bottom_width - bottom.shape[1]
    bottom_start = (bottom_start[0] + change_in_width // 2, bottom_start[1] + change_in_height // 2)

    top = cv2.warpAffine(top, cv2.getRotationMatrix2D((top.shape[1] / 2, top.shape[0] / 2), params[0]['r'], 1), (top.shape[1], top.shape[0]))
    center = cv2.warpAffine(center, cv2.getRotationMatrix2D((center.shape[1] / 2, center.shape[0] / 2), params[1]['r'], 1), (center.shape[1], center.shape[0]))
    bottom = cv2.warpAffine(bottom, cv2.getRotationMatrix2D((bottom.shape[1] / 2, bottom.shape[0] / 2), params[2]['r'], 1), (bottom.shape[1], bottom.shape[0]))

    # canvas = copy_frames(canvas, top, top_start, center, center_start, bottom, bottom_start)
    canvas = copy_frames2(canvas, top, top_start, center, center_start, bottom, bottom_start)


    out.write(canvas)
    progress_bar.update(1)
