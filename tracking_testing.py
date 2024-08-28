import cv2
from Annotation import Annotation
import numpy as np
import tqdm

video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'

available_trackers = ["CSRT", "MIL", "KCF", "DISOpticalFlow", "FarnebackOpticalFlow", "SparsePyrLKOpticalFlow", "KalmanFilter", "DaSiamRPN", "GOTURN", "Nano", "Vit"]
chosen_tracker = "DISOpticalFlow"

# GOTURN is really really slow

# correction step in kalman filter cannot be done using annotations, use background subtractor instead and build tracker class
# optical flow tracking uses only one point for each player, we could use more points to improve tracking
# if the tracking fails (player exiting frame or occlusion or any other problem) we should implement a reinitialization step

# need to build classes for all trackers


cap = cv2.VideoCapture(video)
annotations = Annotation(annotations_folder)
predicted_tracks = []
progress_bar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

if chosen_tracker in ["CSRT", "MIL", "KCF", "DaSiamRPN", "GOTURN", "Nano", "Vit"]:
    if chosen_tracker == "CSRT":
        tracker = cv2.TrackerCSRT
    elif chosen_tracker == "MIL":
        tracker = cv2.TrackerMIL
    elif chosen_tracker == "KCF":
        tracker = cv2.TrackerKCF
    elif chosen_tracker == "DaSiamRPN":
        tracker = cv2.TrackerDaSiamRPN
    elif chosen_tracker == "GOTURN":
        tracker = cv2.TrackerGOTURN
    elif chosen_tracker == "Nano":
        tracker = cv2.TrackerNano
    elif chosen_tracker == "Vit":
        tracker = cv2.TrackerVit

    ret, frame = cap.read()
    annotation = annotations.get_annotation(0)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    tracker_instances = []
    for i in range(len(annotation)):
        tracker_instances.append(tracker.create())
        box = (min(annotation[i].x1, annotation[i].x2), min(annotation[i].y1, annotation[i].y2), abs(annotation[i].x1 - annotation[i].x2), abs(annotation[i].y1 - annotation[i].y2))
        tracker_instances[i].init(frame, box)    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracker_boxes = []
        for i, tracker_instance in enumerate(tracker_instances):
            success, bbox = tracker_instance.update(frame)
            if success:
                tracker_boxes.append(bbox)
                x, y, w, h = [int(i) for i in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        predicted_tracks.append(tracker_boxes)
        cv2.imshow('frame', frame)
        progress_bar.update(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    
    mota, motp, total_switches, total_fragmentations = annotations.evaluate_tracking(predicted_tracks)

    print(f'Multiple Object Tracking Accuracy: {mota}')
    #print(f'Multiple Object Tracking Precision: {motp}')
    print(f'Total ID Switches: {total_switches}')
    print(f'Total Fragmentations: {total_fragmentations}')

if chosen_tracker in ["DISOpticalFlow", "FarnebackOpticalFlow"]:
    if chosen_tracker == "DISOpticalFlow":
        tracker = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST) # PRESET_FAST AND PRESET_MEDIUM
    elif chosen_tracker == "FarnebackOpticalFlow":
        tracker = cv2.FarnebackOpticalFlow.create()

    ret, frame = cap.read()
    flow_seq = []
    height, width = frame.shape[:2]
    division = 3 #change this for optimization
    frame = cv2.resize(frame, (width // division, height // division))

    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('flow magnitude', cv2.WINDOW_NORMAL)
    cv2.namedWindow('bgr', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (width // division, height // division))
        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = tracker.calc(previous_frame, next_frame, None)
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('bgr', bgr)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = np.where(magnitude > 50, 255, 0)
        magnitude = magnitude.astype(np.uint8)
        cv2.imshow('flow magnitude', magnitude)

        cv2.imshow('frame', frame)
        progress_bar.update(1)
        flow_seq.append(flow)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_frame = next_frame

    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    
    displacement_error = annotations.evaluate_optical_flow(flow_seq)
    print(f'Displacement Error: {displacement_error}')

if chosen_tracker in ["SparsePyrLKOpticalFlow"]:
    tracker = cv2.SparsePyrLKOpticalFlow.create()
    tracked_points = []

    ret, frame = cap.read()

    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    annotation = annotations.get_annotation(0)
    previous_points = np.array([[int((gt.x1 + gt.x2) / 2), int((gt.y1 + gt.y2) / 2)] for gt in annotation], dtype=np.float32)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('flow', cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, err = tracker.calc(previous_frame, next_frame, previous_points, None)

        frame_copy = frame.copy()
        frame_copy[:, :, :] = 0

        for i, res in enumerate(status):
            if res == 1:
                prev_x = int(previous_points[i][0])
                prev_y = int(previous_points[i][1])
                next_x = int(next_points[i][0])
                next_y = int(next_points[i][1])

                cv2.line(frame_copy, (prev_x, prev_y), (next_x, next_y), (255, 255, 255), 6)
        
        cv2.imshow('frame', frame)
        cv2.imshow('flow', frame_copy)
        progress_bar.update(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_frame = next_frame
        previous_points = next_points
        tracked_points.append(next_points)
    
    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    
    displacement_error = annotations.evaluate_sparse_optical_flow(tracked_points)
    print(f'Displacement Error: {displacement_error}')

if chosen_tracker in ["KalmanFilter"]:
    ret, frame = cap.read()
    annotation = annotations.get_annotation(0)

    tracker_instances = []
    total_displacement_error = 0
    total_frames = 0

    for i in range(len(annotation)):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03
        
        x, y, x2, y2 = annotation[i].x1, annotation[i].y1, annotation[i].x2, annotation[i].y2
        center_x = int((x + x2) / 2)
        center_y = int((y + y2) / 2)
        kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        
        tracker_instances.append(kalman)
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        for i, tracker_instance in enumerate(tracker_instances):
            prediction = tracker_instance.predict()
            pred_x, pred_y = prediction[0][0], prediction[1][0]

            # Draw the predicted bounding box
            pred_x2, pred_y2 = pred_x + 20, pred_y + 20
            cv2.rectangle(frame, (int(pred_x), int(pred_y)), (int(pred_x2), int(pred_y2)), (0, 255, 0), 2)

            # Get current frame number and the corresponding annotation
            n_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            annotation = annotations.get_annotation(n_frame)
            x, y, x2, y2 = annotation[i].x1, annotation[i].y1, annotation[i].x2, annotation[i].y2
            center_x = int((x + x2) / 2)
            center_y = int((y + y2) / 2)
            measurement = np.array([[center_x], [center_y]], np.float32)

            # Correct the Kalman filter with the actual measurement
            tracker_instance.correct(measurement)

            # Calculate the displacement error
            displacement_error = np.sqrt((pred_x - center_x) ** 2 + (pred_y - center_y) ** 2)
            total_displacement_error += displacement_error

        cv2.imshow('frame', frame)
        progress_bar.update(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()

    # Calculate the average displacement error
    average_displacement_error = total_displacement_error / (total_frames * len(tracker_instances))
    print(f'Average Displacement Error: {average_displacement_error}')
