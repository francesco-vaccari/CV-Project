import cv2
import numpy as np
import histogram as his
import Detection as det
from yolo_model import yolo_model
import math

annotations_folder = 'annotations'
BGSUB = det.BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)

class OpenCVTracker:
    def __init__(self, tracker_name, initial_frame, initial_boxes, show=False):
        available_trackers = ["CSRT", "MIL", "DaSiamRPN", "Nano", "Vit"]
        if tracker_name not in available_trackers:
            raise Exception("Tracker not available. Available trackers are " + str(available_trackers))

        if tracker_name == "CSRT":
            self.tracker = cv2.TrackerCSRT
        elif tracker_name == "MIL":
            self.tracker = cv2.TrackerMIL
        elif tracker_name == "DaSiamRPN":
            self.tracker = cv2.TrackerDaSiamRPN
        elif tracker_name == "Nano":
            self.tracker = cv2.TrackerNano
        elif tracker_name == "Vit":
            self.tracker = cv2.TrackerVit

        self.history_length = 48
        self.tracker_histogram_history = []
        self.tracker_position_history = []
        self.tracker_box_size_history = []

        self.tracker_ready = [0]*len(initial_boxes)
        self.tracker_instances = []
        for i, box in enumerate(initial_boxes):
            self.tracker_instances.append(self.tracker.create())
            self.tracker_instances[i].init(initial_frame, box)
            self.tracker_histogram_history.append([])
            self.tracker_position_history.append([])
            self.tracker_box_size_history.append([])
        for i, box in enumerate(initial_boxes):
            self.update_history(i, box, initial_frame)
        
        self.show = show
    
    def update(self, frame):
        results = []
        boxes = []
        match_results = [False]*len(self.tracker_instances)
        match_boxes = [None]*len(self.tracker_instances)

        for i, tracker_instance in enumerate(self.tracker_instances):
            if self.tracker_ready[i] == 0:
                success, box = tracker_instance.update(frame)
                results.append(success)
                boxes.append(box)
            else:
                results.append(False)
                boxes.append((0,0,0,0))
        
        for i, res in enumerate(results):
            if res:
                if self.is_tracking_lost(boxes[i], frame):
                    self.tracker_ready[i] += 1
                    results[i] = False
                else:
                    self.update_history(i, boxes[i], frame)
            if not res:
                bg = BGSUB.apply(frame)
                bg = det.preprocess(bg)
                motion_boxes = det.extract_boxes(bg)
                match_result, match_box = self.match_boxes(i, motion_boxes, frame)
                match_results[i] = match_result
                match_boxes[i] = match_box
        
        for i, res in enumerate(match_results):
            if res:
                if self.check_overlap(match_boxes[i], boxes, ignore_index=i):
                    self.tracker_instances[i].init(frame, match_boxes[i])
                    self.tracker_ready[i] = 0
                    results[i] = True
                    boxes[i] = match_boxes[i]
                    frame_copy  = frame.copy()
                    if self.show:
                        cv2.rectangle(frame_copy, (match_boxes[i][0], match_boxes[i][1]), (match_boxes[i][0]+match_boxes[i][2], match_boxes[i][1]+match_boxes[i][3]), (255, 0, 0), 4)
                        cv2.imshow('tracker', frame_copy)
                
        
        return results, boxes

    def is_tracking_lost(self, box, frame): # check if the bounding box tracked contains some movement
        bg = BGSUB.apply(frame)
        bg = det.preprocess(bg)
        motion_boxes = det.extract_boxes(bg)

        if len(motion_boxes) == 0:
            return True

        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        for motion_box in motion_boxes:
            x1_m, y1_m, x2_m, y2_m = motion_box
            # if any corner is inside the other box then there is movement
            if x1_m <= x1 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x1 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1 <= x1_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x1_m <= x2 and y1 <= y2_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y2_m <= y2:
                return False
        
        return True

    def update_history(self, tracker_index, box, frame): # update history for tracker tracker_index with color, position and size
        if len(self.tracker_histogram_history[tracker_index]) == self.history_length:
            self.tracker_histogram_history[tracker_index].pop(0)
            self.tracker_position_history[tracker_index].pop(0)
            self.tracker_box_size_history[tracker_index].pop(0)
        self.tracker_histogram_history[tracker_index].append(his.get_histogram(frame, box))
        self.tracker_position_history[tracker_index].append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.tracker_box_size_history[tracker_index].append([box[2], box[3]])

    def match_boxes(self, tracker_index, matching_boxes, frame):
        hist_factor = 0.7
        pos_factor = 0.2
        size_factor = 0.1

        match_score = [0]*len(matching_boxes)
        frames_passed = self.tracker_ready[tracker_index] # frames passed since the tracking has been lost
        for i, box in enumerate(matching_boxes):
            box_histogram = his.get_histogram(frame, box)
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            box_center = [x + w / 2, y + h / 2]

            histogram_comparison = his.compare_histogram_to_history(box_histogram, self.tracker_histogram_history[tracker_index])
            
            position_comparison = 0
            for position in self.tracker_position_history[tracker_index]:
                diff = math.sqrt((box_center[0] - position[0]) ** 2 + (box_center[1] - position[1]) ** 2)
                position_comparison += diff
            position_comparison /= len(self.tracker_position_history[tracker_index])
            position_comparison = position_comparison / (frames_passed + 1)
            
            size_comparison = 0
            for size in self.tracker_box_size_history[tracker_index]:
                diff = math.sqrt((w - size[0]) ** 2 + (h - size[1]) ** 2)
                size_comparison += diff
            size_comparison /= len(self.tracker_box_size_history[tracker_index])

            match_score[i] = hist_factor * histogram_comparison + pos_factor * position_comparison + size_factor * size_comparison
        
        if len(matching_boxes) == 0:
            return False, (0,0,0,0)
        
        best_match = np.argmin(match_score)
        x, y, x2, y2 = matching_boxes[best_match]
        return True, [x, y, x2-x, y2-y]

    def check_overlap(self, box, boxes, ignore_index, overlap_thresh=0.3):
        x1, y1, w1, h1 = box
        box_area = w1 * h1

        for i, other_box in enumerate(boxes):
            if i == ignore_index:
                continue
            x2, y2, w2, h2 = other_box
            other_box_area = w2 * h2

            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersect_area = x_overlap * y_overlap

            min_area = min(box_area, other_box_area)
            
            try:
                overlap_ratio = intersect_area / min_area
            except ZeroDivisionError:
                overlap_ratio = 0

            if overlap_ratio > overlap_thresh:
                return False

        return True



class DenseOpticalFlowTracker:
    def __init__(self, tracker_name, frame, initial_boxes, points_sampling='gaussian25', show=False):
        available_trackers = ["DISOpticalFlow", "FarnebackOpticalFlow"]
        if tracker_name not in available_trackers:
            raise Exception("Tracker not available. Available trackers are " + str(available_trackers))

        available_points_sampling = ['gaussian9', 'gaussian16', 'gaussian25', 'center', 'random9', 'random16', 'random25']
        if points_sampling not in available_points_sampling:
            raise Exception("Points sampling not available. Available points sampling are " + str(available_points_sampling))
        
        if tracker_name == "DISOpticalFlow":
            self.tracker = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST) # PRESET_ULTRAFAST, PRESET_FAST AND PRESET_MEDIUM
        elif tracker_name == "FarnebackOpticalFlow":
            self.tracker = cv2.FarnebackOpticalFlow.create()
        
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame_color = frame
        self.boxes = []
        for box in initial_boxes:
            x, y, w, h = box
            self.boxes.append([x, y, w, h])

        self.points_sampling = points_sampling
        self.points = []
        for i in range(len(self.boxes)):
            self.points.append(self.sample_points(self.boxes[i]))
        
        self.history_length = 36
        self.tracker_histogram_history = []
        self.tracker_position_history = []
        self.tracker_box_size_history = []

        self.tracker_ready = [0]*len(initial_boxes)
        self.tracker_instances = []
        for i, box in enumerate(initial_boxes):
            self.tracker_histogram_history.append([])
            self.tracker_position_history.append([])
            self.tracker_box_size_history.append([])
        for i, box in enumerate(initial_boxes):
            self.update_history(i, box, frame)

        self.show = show
    
    def update_history(self, tracker_index, box, frame): # update history for tracker tracker_index with color, position and size
        if len(self.tracker_histogram_history[tracker_index]) == self.history_length:
            self.tracker_histogram_history[tracker_index].pop(0)
            self.tracker_position_history[tracker_index].pop(0)
            self.tracker_box_size_history[tracker_index].pop(0)
        self.tracker_histogram_history[tracker_index].append(his.get_histogram(frame, box))
        self.tracker_position_history[tracker_index].append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.tracker_box_size_history[tracker_index].append([box[2], box[3]])
    
    def sample_points(self, box, single_point_resample=False):
        points = []
        box = box[0], box[1], box[2]+box[0], box[3]+box[1]
        min_x = min(box[0], box[2])
        max_x = max(box[0], box[2])
        min_y = min(box[1], box[3])
        max_y = max(box[1], box[3])
        box = [min_x+1, min_y+1, max_x-1, max_y-1]
        if self.points_sampling == 'center':
            point = [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]
            points.append(point)
            if single_point_resample:
                return point
        elif self.points_sampling == 'random9':
            for i in range(9):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'random16':
            for i in range(16):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'random25':
            for i in range(25):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'gaussian9' or self.points_sampling == 'gaussian16' or self.points_sampling == 'gaussian25':
            num_points = int(self.points_sampling[8:])
            mean_x = (box[0] + box[2]) / 2
            mean_y = (box[1] + box[3]) / 2
            std_x = (math.fabs(box[0] - box[2]) + math.fabs(box[1] - box[3])) / 6
            std_y = (math.fabs(box[0] - box[2]) + math.fabs(box[1] - box[3])) / 6
            for i in range(num_points):
                sampled_x = np.random.normal(loc=mean_x, scale=std_x)
                sampled_y = np.random.normal(loc=mean_y, scale=std_y)
                sampled_x = np.clip(sampled_x, box[0], box[2])
                sampled_y = np.clip(sampled_y, box[1], box[3])
                points.append([int(sampled_x), int(sampled_y)])
                if single_point_resample:
                    return [int(sampled_x), int(sampled_y)]

        return points
    
    def update(self, frame):
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame_color = frame
        frame_copy = frame.copy()
        flow = self.tracker.calc(self.previous_frame, self.next_frame, None)

        match_results = [False]*len(self.boxes)
        match_boxes = [None]*len(self.boxes)

        results = self.update_boxes(flow, frame_copy)

        for i, res in enumerate(results):
            if res:
                if self.is_tracking_lost(self.boxes[i], frame):
                    self.tracker_ready[i] += 1
                    results[i] = False
                else:
                    self.update_history(i, self.boxes[i], frame)
            if not res:
                bg = BGSUB.apply(frame)
                bg = det.preprocess(bg)
                motion_boxes = det.extract_boxes(bg)
                match_result, match_box = self.match_boxes(i, motion_boxes, frame)
                match_results[i] = match_result
                match_boxes[i] = match_box
        
        for i, res in enumerate(match_results):
            if res:
                if self.check_overlap(match_boxes[i], self.boxes, ignore_index=i):
                    self.tracker_ready[i] = 0
                    results[i] = True
                    self.boxes[i] = match_boxes[i]
                    for j, point in enumerate(self.points[i]):
                        self.points[i][j] = self.sample_points(self.boxes[i], single_point_resample=True)
                    if self.show:
                        cv2.rectangle(frame_copy, (match_boxes[i][0], match_boxes[i][1]), (match_boxes[i][0]+match_boxes[i][2], match_boxes[i][1]+match_boxes[i][3]), (255, 0, 0), 4)

        self.previous_frame = self.next_frame
        
        if self.show:
            cv2.imshow('tracker', frame_copy)
        
        return results, self.boxes

    def update_boxes(self, flow, frame_copy):
        success = [False for i in range(len(self.boxes))]
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        for i, points in enumerate(self.points):
            if self.tracker_ready[i] == 0:
                # apply the flow to the points
                for j, point in enumerate(points):
                    x, y = point
                    magnitude_point = magnitude[y, x]
                    angle_point = angle[y, x]
                    delta_x = magnitude_point * math.cos(angle_point)
                    delta_y = magnitude_point * math.sin(angle_point)
                    new_x = x + delta_x
                    new_y = y + delta_y
                    new_x = np.clip(new_x, 0, self.next_frame.shape[1]-1)
                    new_y = np.clip(new_y, 0, self.next_frame.shape[0]-1)
                    self.points[i][j] = [int(new_x), int(new_y)]

                # find the minimal box that contains all the points
                x_min = self.next_frame.shape[1] + 1
                y_min = self.next_frame.shape[0] + 1
                x_max = -1
                y_max = -1
                center = [0, 0]
                for j, point in enumerate(self.points[i]):
                    x, y = point
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    center[0] += x
                    center[1] += y
                    if self.show:
                        cv2.circle(frame_copy, (point[0], point[1]), 4, (0, 255, 0), -1)
                center[0] /= len(self.points[i])
                center[1] /= len(self.points[i])
                if self.show:
                    cv2.circle(frame_copy, (int(center[0]), int(center[1])), 8, (255, 0, 0), -1)
                    cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)

                # find the motion detector box with the center closest to the edges of the box
                # but not any box is fine, it must contain the center of the points found
                bg = BGSUB.apply(self.next_frame_color)
                bg = det.preprocess(bg)
                new_boxes = det.extract_boxes(bg)
                box_index = -1
                min_distance = -1
                for j, new_box in enumerate(new_boxes):
                    new_center = [(new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2]
                    distance = math.sqrt((new_center[0] - center[0]) ** 2 + (new_center[1] - center[1]) ** 2)
                    if distance < min_distance or min_distance == -1:
                        if new_box[0] <= center[0] <= new_box[2] and new_box[1] <= center[1] <= new_box[3]:
                            min_distance = distance
                            box_index = j

                if box_index == -1:
                    # means no motion detector box was suitable
                    success[i] = False
                
                if box_index != -1:
                    # means that a motion detector box was found
                    success[i] = True

                    # the new box size will be the mean between the surrounding box size, the previous box size and the motion detector box size
                    x1 = self.boxes[i][0]
                    y1 = self.boxes[i][1]
                    x2 = self.boxes[i][2] + x1
                    y2 = self.boxes[i][3] + y1
                    new_x1 = new_boxes[box_index][0]
                    new_y1 = new_boxes[box_index][1]
                    new_x2 = new_boxes[box_index][2]
                    new_y2 = new_boxes[box_index][3]
                    if self.show:
                        cv2.rectangle(frame_copy, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 4)
                    motion_box_factor = 0.55
                    points_box_factor = 0.45
                    old_box_factor = 0.0
                    mean_x1 = int(points_box_factor*x_min + old_box_factor*x1 + motion_box_factor*new_x1)
                    mean_y1 = int(points_box_factor*y_min + old_box_factor*y1 + motion_box_factor*new_y1)
                    mean_x2 = int(points_box_factor*x_max + old_box_factor*x2 + motion_box_factor*new_x2)
                    mean_y2 = int(points_box_factor*y_max + old_box_factor*y2 + motion_box_factor*new_y2)
                    if self.show:
                        cv2.rectangle(frame_copy, (mean_x1, mean_y1), (mean_x2, mean_y2), (0, 255, 0), 4)

                    self.boxes[i] = [mean_x1, mean_y1, mean_x2-mean_x1, mean_y2-mean_y1]
                
                    # resample the points that are not inside of the box
                    for j, point in enumerate(self.points[i]):
                        x, y = point
                        if not mean_x1 <= x <= mean_x2 or not mean_y1 <= y <= mean_y2:
                            self.points[i][j] = self.sample_points(self.boxes[i], single_point_resample=True)

        return success

    def is_tracking_lost(self, box, frame): # check if the bounding box tracked contains some movement
        bg = BGSUB.apply(frame)
        bg = det.preprocess(bg)
        motion_boxes = det.extract_boxes(bg)

        if len(motion_boxes) == 0:
            return True

        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        for motion_box in motion_boxes:
            x1_m, y1_m, x2_m, y2_m = motion_box
            # if any corner is inside the other box then there is movement
            if x1_m <= x1 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x1 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1 <= x1_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x1_m <= x2 and y1 <= y2_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y2_m <= y2:
                return False
        
        return True

    def match_boxes(self, tracker_index, matching_boxes, frame):
        hist_factor = 0.7
        pos_factor = 0.2
        size_factor = 0.1

        match_score = [0]*len(matching_boxes)
        frames_passed = self.tracker_ready[tracker_index] # frames passed since the tracking has been lost
        for i, box in enumerate(matching_boxes):
            box_histogram = his.get_histogram(frame, box)
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            box_center = [x + w / 2, y + h / 2]

            histogram_comparison = his.compare_histogram_to_history(box_histogram, self.tracker_histogram_history[tracker_index])
            
            position_comparison = 0
            for position in self.tracker_position_history[tracker_index]:
                diff = math.sqrt((box_center[0] - position[0]) ** 2 + (box_center[1] - position[1]) ** 2)
                position_comparison += diff
            position_comparison /= len(self.tracker_position_history[tracker_index])
            position_comparison = position_comparison / (frames_passed + 1)
            
            size_comparison = 0
            for size in self.tracker_box_size_history[tracker_index]:
                diff = math.sqrt((w - size[0]) ** 2 + (h - size[1]) ** 2)
                size_comparison += diff
            size_comparison /= len(self.tracker_box_size_history[tracker_index])

            match_score[i] = hist_factor * histogram_comparison + pos_factor * position_comparison + size_factor * size_comparison
        
        if len(matching_boxes) == 0:
            return False, (0,0,0,0)
        
        best_match = np.argmin(match_score)
        x, y, x2, y2 = matching_boxes[best_match]
        return True, [x, y, x2-x, y2-y]

    def check_overlap(self, box, boxes, ignore_index, overlap_thresh=0.3):
        x1, y1, w1, h1 = box
        box_area = w1 * h1

        for i, other_box in enumerate(boxes):
            if i == ignore_index:
                continue
            x2, y2, w2, h2 = other_box
            other_box_area = w2 * h2

            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersect_area = x_overlap * y_overlap

            min_area = min(box_area, other_box_area)

            try:
                overlap_ratio = intersect_area / min_area
            except ZeroDivisionError:
                overlap_ratio = 0

            if overlap_ratio > overlap_thresh:
                return False

        return True



class PyrLKOpticalFlowTracker:
    def __init__(self, frame, initial_boxes, points_sampling='gaussian25', show=False):
        available_points_sampling = ['gaussian9', 'gaussian16', 'gaussian25', 'center', 'random9', 'random16', 'random25']
        if points_sampling not in available_points_sampling:
            raise Exception("Points sampling not available. Available points sampling are " + str(available_points_sampling))
        
        self.tracker = cv2.SparsePyrLKOpticalFlow.create()
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame_color = frame
        self.boxes = []
        for box in initial_boxes:
            x, y, w, h = box
            self.boxes.append([x, y, w, h])
        
        self.points_sampling = points_sampling
        self.previous_points = []
        self.next_points = []
        for i in range(len(self.boxes)):
            self.previous_points.append(self.sample_points(self.boxes[i]))
            self.next_points.append(self.previous_points[i])

        self.history_length = 36
        self.tracker_histogram_history = []
        self.tracker_position_history = []
        self.tracker_box_size_history = []

        self.tracker_ready = [0]*len(initial_boxes)
        self.tracker_instances = []
        for i, box in enumerate(initial_boxes):
            self.tracker_histogram_history.append([])
            self.tracker_position_history.append([])
            self.tracker_box_size_history.append([])
        for i, box in enumerate(initial_boxes):
            self.update_history(i, box, frame)
        
        self.show = show

    def update_history(self, tracker_index, box, frame): # update history for tracker tracker_index with color, position and size
        if len(self.tracker_histogram_history[tracker_index]) == self.history_length:
            self.tracker_histogram_history[tracker_index].pop(0)
            self.tracker_position_history[tracker_index].pop(0)
            self.tracker_box_size_history[tracker_index].pop(0)
        self.tracker_histogram_history[tracker_index].append(his.get_histogram(frame, box))
        self.tracker_position_history[tracker_index].append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.tracker_box_size_history[tracker_index].append([box[2], box[3]])
    
    def sample_points(self, box, single_point_resample=False):
        points = []
        box = box[0], box[1], box[2]+box[0], box[3]+box[1]
        min_x = min(box[0], box[2])
        max_x = max(box[0], box[2])
        min_y = min(box[1], box[3])
        max_y = max(box[1], box[3])
        box = [min_x+1, min_y+1, max_x-1, max_y-1]
        if self.points_sampling == 'center':
            point = [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]
            points.append(point)
            if single_point_resample:
                return point
        elif self.points_sampling == 'random9':
            for i in range(9):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'random16':
            for i in range(16):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'random25':
            for i in range(25):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'gaussian9' or self.points_sampling == 'gaussian16' or self.points_sampling == 'gaussian25':
            num_points = int(self.points_sampling[8:])
            mean_x = (box[0] + box[2]) / 2
            mean_y = (box[1] + box[3]) / 2
            std_x = (math.fabs(box[0] - box[2]) + math.fabs(box[1] - box[3])) / 6
            std_y = (math.fabs(box[0] - box[2]) + math.fabs(box[1] - box[3])) / 6
            for i in range(num_points):
                sampled_x = np.random.normal(loc=mean_x, scale=std_x)
                sampled_y = np.random.normal(loc=mean_y, scale=std_y)
                sampled_x = np.clip(sampled_x, box[0], box[2])
                sampled_y = np.clip(sampled_y, box[1], box[3])
                points.append([int(sampled_x), int(sampled_y)])
                if single_point_resample:
                    return [int(sampled_x), int(sampled_y)]

        return points

    def update(self, frame):
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame_color = frame
        frame_copy = frame.copy()

        match_results = [False]*len(self.boxes)
        match_boxes = [None]*len(self.boxes)
        results = [False for i in range(len(self.boxes))]
        for i in range(len(self.boxes)):
            input_points = np.array(self.previous_points[i], dtype=np.float32)
            self.next_points[i], success, error = self.tracker.calc(self.previous_frame, self.next_frame, input_points, None)
            self.next_points[i], results[i] = self.update_boxes(i, self.next_points, success, error, frame_copy)

        for i, res in enumerate(results):
            if res:
                if self.is_tracking_lost(self.boxes[i], frame):
                    self.tracker_ready[i] += 1
                    results[i] = False
                else:
                    self.update_history(i, self.boxes[i], frame)
            if not res:
                bg = BGSUB.apply(frame)
                bg = det.preprocess(bg)
                motion_boxes = det.extract_boxes(bg)
                match_result, match_box = self.match_boxes(i, motion_boxes, frame)
                match_results[i] = match_result
                match_boxes[i] = match_box
        
        for i, res in enumerate(match_results):
            if res:
                if self.check_overlap(match_boxes[i], self.boxes, ignore_index=i):
                    self.tracker_ready[i] = 0
                    results[i] = True
                    self.boxes[i] = match_boxes[i]
                    for j, point in enumerate(self.next_points[i]):
                        self.next_points[i][j] = self.sample_points(self.boxes[i], single_point_resample=True)
                    if self.show:
                        cv2.rectangle(frame_copy, (match_boxes[i][0], match_boxes[i][1]), (match_boxes[i][0]+match_boxes[i][2], match_boxes[i][1]+match_boxes[i][3]), (255, 0, 0), 4)

        self.previous_frame = self.next_frame
        self.previous_points = self.next_points
        
        if self.show:
            cv2.imshow('tracker', frame_copy)
        
        return results, self.boxes

    def is_tracking_lost(self, box, frame): # check if the bounding box tracked contains some movement
        bg = BGSUB.apply(frame)
        bg = det.preprocess(bg)
        motion_boxes = det.extract_boxes(bg)

        if len(motion_boxes) == 0:
            return True

        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        for motion_box in motion_boxes:
            x1_m, y1_m, x2_m, y2_m = motion_box
            # if any corner is inside the other box then there is movement
            if x1_m <= x1 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x1 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1 <= x1_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x1_m <= x2 and y1 <= y2_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y2_m <= y2:
                return False
        
        return True

    def update_boxes(self, index, next_points, flow_success, flow_error, frame_copy):
        success = False
        next_points = next_points[index]
        len_valid_points = sum([flow_success[j][0] for j in range(len(flow_success))])

        if self.tracker_ready[index] == 0 and len_valid_points > 0:
            # find the minimal box that contains all the valid points
            x_min = self.next_frame.shape[1] + 1
            y_min = self.next_frame.shape[0] + 1
            x_max = -1
            y_max = -1
            center = [0, 0]
            for j, point in enumerate(next_points):
                if flow_success[j][0] == 1:
                    x, y = int(point[0]), int(point[1])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    center[0] += x
                    center[1] += y
                    if self.show:
                        cv2.circle(frame_copy, (x, y), 4, (0, 255, 0), -1)
            center[0] /= len_valid_points
            center[1] /= len_valid_points
            if self.show:
                cv2.circle(frame_copy, (int(center[0]), int(center[1])), 8, (255, 0, 0), -1)
                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)

            # find the motion detector box with the center closest to the edges of the box
            # but not any box is fine, it must contain the center of the points found
            bg = BGSUB.apply(self.next_frame_color)
            bg = det.preprocess(bg)
            new_boxes = det.extract_boxes(bg)
            box_index = -1
            min_distance = -1
            for j, new_box in enumerate(new_boxes):
                new_center = [(new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2]
                distance = math.sqrt((new_center[0] - center[0]) ** 2 + (new_center[1] - center[1]) ** 2)
                if distance < min_distance or min_distance == -1:
                    if new_box[0] <= center[0] <= new_box[2] and new_box[1] <= center[1] <= new_box[3]:
                        min_distance = distance
                        box_index = j

            if box_index == -1:
                # means no motion detector box was suitable
                success = False
            
            if box_index != -1:
                # means that a motion detector box was found
                success = True

                # the new box size will be the mean between the surrounding box size, the previous box size and the motion detector box size
                x1 = self.boxes[index][0]
                y1 = self.boxes[index][1]
                x2 = self.boxes[index][2] + x1
                y2 = self.boxes[index][3] + y1
                new_x1 = new_boxes[box_index][0]
                new_y1 = new_boxes[box_index][1]
                new_x2 = new_boxes[box_index][2]
                new_y2 = new_boxes[box_index][3]
                if self.show:
                    cv2.rectangle(frame_copy, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 4)
                motion_box_factor = 0.55
                points_box_factor = 0.45
                old_box_factor = 0.0
                mean_x1 = int(points_box_factor*x_min + old_box_factor*x1 + motion_box_factor*new_x1)
                mean_y1 = int(points_box_factor*y_min + old_box_factor*y1 + motion_box_factor*new_y1)
                mean_x2 = int(points_box_factor*x_max + old_box_factor*x2 + motion_box_factor*new_x2)
                mean_y2 = int(points_box_factor*y_max + old_box_factor*y2 + motion_box_factor*new_y2)
                if self.show:
                    cv2.rectangle(frame_copy, (mean_x1, mean_y1), (mean_x2, mean_y2), (0, 255, 0), 4)

                self.boxes[index] = [mean_x1, mean_y1, mean_x2-mean_x1, mean_y2-mean_y1]
            
                # resample the points that are not inside of the box and the points that were not valid
                for j, point in enumerate(next_points):
                    x, y = point
                    if not mean_x1 <= x <= mean_x2 or not mean_y1 <= y <= mean_y2 or flow_success[j][0] == 0:
                        next_points[j] = self.sample_points(self.boxes[index], single_point_resample=True)

        return next_points, success

    def match_boxes(self, tracker_index, matching_boxes, frame):
        hist_factor = 0.7
        pos_factor = 0.2
        size_factor = 0.1

        match_score = [0]*len(matching_boxes)
        frames_passed = self.tracker_ready[tracker_index] # frames passed since the tracking has been lost
        for i, box in enumerate(matching_boxes):
            box_histogram = his.get_histogram(frame, box)
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            box_center = [x + w / 2, y + h / 2]

            histogram_comparison = his.compare_histogram_to_history(box_histogram, self.tracker_histogram_history[tracker_index])
            
            position_comparison = 0
            for position in self.tracker_position_history[tracker_index]:
                diff = math.sqrt((box_center[0] - position[0]) ** 2 + (box_center[1] - position[1]) ** 2)
                position_comparison += diff
            position_comparison /= len(self.tracker_position_history[tracker_index])
            position_comparison = position_comparison / (frames_passed + 1)
            
            size_comparison = 0
            for size in self.tracker_box_size_history[tracker_index]:
                diff = math.sqrt((w - size[0]) ** 2 + (h - size[1]) ** 2)
                size_comparison += diff
            size_comparison /= len(self.tracker_box_size_history[tracker_index])

            match_score[i] = hist_factor * histogram_comparison + pos_factor * position_comparison + size_factor * size_comparison
        
        if len(matching_boxes) == 0:
            return False, (0,0,0,0)
        
        best_match = np.argmin(match_score)
        x, y, x2, y2 = matching_boxes[best_match]
        return True, [x, y, x2-x, y2-y]

    def check_overlap(self, box, boxes, ignore_index, overlap_thresh=0.3):
        x1, y1, w1, h1 = box
        box_area = w1 * h1

        for i, other_box in enumerate(boxes):
            if i == ignore_index:
                continue
            x2, y2, w2, h2 = other_box
            other_box_area = w2 * h2

            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersect_area = x_overlap * y_overlap

            min_area = min(box_area, other_box_area)

            try:
                overlap_ratio = intersect_area / min_area
            except ZeroDivisionError:
                overlap_ratio = 0

            if overlap_ratio > overlap_thresh:
                return False

        return True



class KalmanFilterTracker:
    def __init__(self, frame, initial_boxes, show=False):
        self.tracker_ready = [0]*len(initial_boxes)
        self.boxes = initial_boxes
        self.tracker_instances = [None]*len(initial_boxes)

        self.history_length = 36
        self.tracker_histogram_history = []
        self.tracker_position_history = []
        self.tracker_box_size_history = []
        for i, box in enumerate(initial_boxes):
            self.tracker_histogram_history.append([])
            self.tracker_position_history.append([])
            self.tracker_box_size_history.append([])

        for i, box in enumerate(initial_boxes):
            self.reset_tracker_instance(i, box, frame)
        
        self.yolo = yolo_model()
        self.frame = frame
        self.yolo_box_measure_error_threshold = 0.75 # x means the box matched can be distant at most x of the box size
        
        self.n_frame_to_reset = 36

        self.show = show
    
    def reset_tracker_instance(self, tracker_index, box, frame):
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
        
        x, y, w, h = box
        center_x = x + w / 2
        center_y = y + h / 2
        kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        
        self.tracker_instances[tracker_index] = kalman

        self.tracker_box_size_history[tracker_index] = [[w, h]]
        self.tracker_position_history[tracker_index] = [[center_x, center_y]]
        self.tracker_histogram_history[tracker_index] = [his.get_histogram(frame, box)]
    
    def update(self, frame):
        frame_copy = frame.copy()
        results = [False]*len(self.tracker_instances)
        match_results = [False]*len(self.tracker_instances)
        match_boxes = [None]*len(self.tracker_instances)

        yolo_boxes = self.yolo.predict(frame)

        for i, kalman in enumerate(self.tracker_instances):
            self.tracker_ready[i] += 1
            if self.tracker_ready[i] <= self.n_frame_to_reset:
                prediction = kalman.predict()

                pred_x, pred_y = prediction[0][0], prediction[1][0]
                if self.show:
                    cv2.circle(frame_copy, (int(pred_x), int(pred_y)), 4, (255, 0, 0), -1)

                res, obs, box = self.get_observation(yolo_boxes, (pred_x, pred_y), self.boxes[i])

                if res:
                    measurement = np.array([[obs[0]], [obs[1]]], np.float32)
                    kalman.correct(measurement)

                    results[i] = True
                    self.tracker_ready[i] = 0
                    self.boxes[i] = box

                    self.update_history(i, box, frame)

                    if self.show:
                        cv2.rectangle(frame_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 4)
                else:
                    # skip the correction step
                    
                    results[i] = True

                    # move the box around the center predicted with same size of before
                    prev_w = self.boxes[i][2]
                    prev_h = self.boxes[i][3]
                    new_x = int(pred_x - prev_w / 2)
                    new_y = int(pred_y - prev_h / 2)
                    box = [new_x, new_y, prev_w, prev_h]
                    self.boxes[i] = box

                    if self.show:
                        cv2.rectangle(frame_copy, (new_x, new_y), (new_x+prev_w, new_y+prev_h), (0, 0, 255), 4)
            else:
                bg = BGSUB.apply(frame)
                bg = det.preprocess(bg)
                motion_boxes = det.extract_boxes(bg)
                match_result, match_box = self.match_boxes(i, motion_boxes, frame)
                match_results[i] = match_result
                match_boxes[i] = match_box
        
        for i, res in enumerate(match_results):
            if res:
                if self.check_overlap(match_boxes[i], self.boxes, ignore_index=i):
                    self.tracker_ready[i] = 0
                    results[i] = True
                    self.boxes[i] = match_boxes[i]
                    self.reset_tracker_instance(i, match_boxes[i], frame)
                    if self.show:
                        cv2.rectangle(frame_copy, (match_boxes[i][0], match_boxes[i][1]), (match_boxes[i][0]+match_boxes[i][2], match_boxes[i][1]+match_boxes[i][3]), (0, 255, 255), 4)

        if self.show:
            cv2.imshow('tracker', frame_copy)
        return results, self.boxes
    
    def update_history(self, tracker_index, box, frame):
        if len(self.tracker_histogram_history[tracker_index]) == self.history_length:
            self.tracker_histogram_history[tracker_index].pop(0)
            self.tracker_position_history[tracker_index].pop(0)
            self.tracker_box_size_history[tracker_index].pop(0)
        self.tracker_histogram_history[tracker_index].append(his.get_histogram(frame, box))
        self.tracker_position_history[tracker_index].append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.tracker_box_size_history[tracker_index].append([box[2], box[3]])

    def get_observation(self, boxes, center, old_box):
        if len(boxes) == 0:
            return False, (0,0), (0,0,0,0)
        
        old_box_size = (old_box[2] + old_box[3]) / 2

        min_distance = -1
        best_box = None
        best_box_center = None
        for box in boxes:
            x1, y1, x2, y2 = box
            box_center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            distance = math.sqrt((center[0] - box_center[0]) ** 2 + (center[1] - box_center[1]) ** 2)
            if distance < min_distance or min_distance == -1:
                min_distance = distance
                best_box = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                best_box_center = box_center

        if min_distance > old_box_size * self.yolo_box_measure_error_threshold:
            return False, (0,0), (0,0,0,0)
        
        return True, best_box_center, best_box
    
    def match_boxes(self, tracker_index, matching_boxes, frame):
        hist_factor = 0.7
        pos_factor = 0.2
        size_factor = 0.1

        match_score = [0]*len(matching_boxes)
        frames_passed = self.tracker_ready[tracker_index] # frames passed since the tracking has been lost
        for i, box in enumerate(matching_boxes):
            box_histogram = his.get_histogram(frame, box)
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            box_center = [x + w / 2, y + h / 2]

            histogram_comparison = his.compare_histogram_to_history(box_histogram, self.tracker_histogram_history[tracker_index])
            
            position_comparison = 0
            for position in self.tracker_position_history[tracker_index]:
                diff = math.sqrt((box_center[0] - position[0]) ** 2 + (box_center[1] - position[1]) ** 2)
                position_comparison += diff
            position_comparison /= len(self.tracker_position_history[tracker_index])
            position_comparison = position_comparison / (frames_passed + 1)
            
            size_comparison = 0
            for size in self.tracker_box_size_history[tracker_index]:
                diff = math.sqrt((w - size[0]) ** 2 + (h - size[1]) ** 2)
                size_comparison += diff
            size_comparison /= len(self.tracker_box_size_history[tracker_index])

            match_score[i] = hist_factor * histogram_comparison + pos_factor * position_comparison + size_factor * size_comparison
        
        if len(matching_boxes) == 0:
            return False, (0,0,0,0)
        
        best_match = np.argmin(match_score)
        x, y, x2, y2 = matching_boxes[best_match]
        return True, [x, y, x2-x, y2-y]

    def check_overlap(self, box, boxes, ignore_index, overlap_thresh=0.3):
        x1, y1, w1, h1 = box
        box_area = w1 * h1

        for i, other_box in enumerate(boxes):
            if i == ignore_index:
                continue
            x2, y2, w2, h2 = other_box
            other_box_area = w2 * h2

            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersect_area = x_overlap * y_overlap

            min_area = min(box_area, other_box_area)

            try:
                overlap_ratio = intersect_area / min_area
            except ZeroDivisionError:
                overlap_ratio = 0

            if overlap_ratio > overlap_thresh:
                return False

        return True