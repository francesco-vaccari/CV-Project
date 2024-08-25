import os
import numpy as np
from itertools import combinations

class FrameAnnotation:
    def __init__(self, frame, team, player, x1, y1, x2, y2):
        self.frame = frame
        self.team = team
        self.player = player
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

class Annotation:
    def __init__(self, annotations_folder, n_teams = 2, n_players_per_team = 6):
        annotations_files = [os.path.join(annotations_folder, file) for file in os.listdir(annotations_folder) if file.endswith('.label')]
        self.annotations = []
        for i in range(n_teams):
            self.annotations.append([])
            for j in range(n_players_per_team):
                self.annotations[i].append([])
        self.load_annotations(annotations_files)
    
    def load_annotations(self, annotations_files):
        for file in annotations_files:
            try:
                team, player = file.split('_')[1], file.split('_')[2].split('.')[0]
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        frame, x1, y1, x2, y2 = line.strip().split(',')
                        frame = int(frame)-1
                        x1, y1, x2, y2 = min(int(x1), int(x2)), min(int(y1), int(y2)), max(int(x1), int(x2)), max(int(y1), int(y2))
                        self.annotations[int(team)-1][int(player)-1].append(FrameAnnotation(frame, team, player, x1, y1, x2, y2)) # frame number is the same as the index
            except Exception as e:
                # the annotation is for the ball, not for a player
                pass
                

    def get_annotation(self, n_frame):
        annotations = []
        for team in self.annotations:
            for player in team:
                annotations.append(player[n_frame])
        return annotations

    def evaluate(self, boxes, n_frame, threshold=0.9):
        if len(boxes) == 0:
            return 0, 0, []
        
        annotations = self.get_annotation(n_frame)
        tp = 0 # true positives
        fn = 0 # false negatives
        fp = 0 # false positives
        
        matches = {}
        index = 0
        for i, gt in enumerate(annotations):
            x1, y1, x2, y2 = gt.x1, gt.y1, gt.x2, gt.y2
            for j, pred in enumerate(boxes):
                pred_x1, pred_y1, pred_x2, pred_y2 = pred
                iou = self.calculate_iou([x1, y1, x2, y2], [pred_x1, pred_y1, pred_x2, pred_y2])
                if iou > threshold:
                    matches[index] = (i, j)
                    index += 1
        
        if index == 0:
            return 0, 0, []

        best_n_matches = 0
        annotations_used = []
        preds_used = []
        possible_matches = self.all_combinations(list(range(index)))
        for possible_match in possible_matches:
            n_matches = 0
            annotations_indexes = []
            preds_indexes = []
            for match in possible_match:
                i, j = matches[match]
                if i not in annotations_indexes and j not in preds_indexes:
                    n_matches += 1
                    annotations_indexes.append(i)
                    preds_indexes.append(j)
            
            if n_matches > best_n_matches:
                best_n_matches = n_matches
                annotations_used = annotations_indexes
                preds_used = preds_indexes
        
        tp = best_n_matches
        fn = len(annotations) - best_n_matches
        fp = len(boxes) - best_n_matches

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0

        return precision, recall, preds_used

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        try:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        except ZeroDivisionError:
            iou = 0
        
        return iou

    def all_combinations(self, nums):
        result = []
        for r in range(len(nums) + 1):
            result.extend([list(comb) for comb in combinations(nums, r)])
        return result
    
    def evaluate_tracking(self, predicted_tracks, threshold = 0.5):
        mota = 0
        motp = 0
        total_objects = 0
        total_matches = 0
        total_switches = 0
        total_fragmentations = 0
        total_iou = 0
        total_fp = 0
        total_fn = 0

        for frame in range(len(predicted_tracks)):
            precision, recall, preds_used = self.evaluate(predicted_tracks[frame], frame, threshold)
            total_objects += len(self.get_annotation(frame))
            total_matches += len(preds_used)

            #ID Switches and Fragmentation calculation
            if frame > 1:
                prev_frame = frame - 1
                prev_annotations = self.get_annotation(prev_frame)
                curr_annotations = self.get_annotation(frame)

                id_switches, fragmentations = self.calculate_id_switches_and_fragmentations(prev_annotations, curr_annotations, preds_used, predicted_tracks[prev_frame], predicted_tracks[frame])
                total_switches += id_switches
                total_fragmentations += fragmentations

            tp = len(preds_used)  
            fn = len(self.get_annotation(frame)) - tp  
            fp = len(predicted_tracks[frame]) - tp  

            total_fp += fp
            total_fn += fn
            
            for i, j in enumerate(preds_used):
                annotation_box = [self.get_annotation(frame)[i].x1, self.get_annotation(frame)[i].y1, self.get_annotation(frame)[i].x2, self.get_annotation(frame)[i].y2]
                x1, y1, x2, y2 = predicted_tracks[frame][j]
                predicted_box = [x1, y1, x2 + x1, y2 + y1]
                total_iou += self.calculate_iou(annotation_box, predicted_box)

        try:
            mota += 1 - (fp + fn + total_switches) / total_objects
        except ZeroDivisionError:
            mota += 0
            
            
        try:
            motp += total_iou / total_matches
        except ZeroDivisionError:
            motp += 0
        
        return mota, motp, total_switches, total_fragmentations
    
    def calculate_id_switches_and_fragmentations(self, prev_annotations, curr_annotations, preds_used, prev_tracks, curr_tracks):
        id_switches = 0
        fragmentations = 0

        prev_dict = {i: j for i, j in enumerate(prev_annotations)}
        curr_dict = {i: j for i, j in enumerate(curr_annotations)}

        for i, j in zip(prev_dict.keys(), preds_used):
            prev_id = prev_dict[i].player
            curr_id = curr_dict[j].player

            if prev_id != curr_id:
                id_switches += 1

            if i not in preds_used:
                fragmentations += 1

        return id_switches, fragmentations
    
    def evaluate_optical_flow(self, flow_sequence):
        total_displacement_error = 0
        total_frames = len(flow_sequence)

        for frame in range(total_frames):
            flow = flow_sequence[frame]
            annotations = self.get_annotation(frame)

            for i in range(12):
                x1, y1, x2, y2 = annotations[i].x1, annotations[i].y1, annotations[i].x2, annotations[i].y2
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi_flow = flow[y1:y2, x1:x2]
                
                if roi_flow.size == 0:
                    #print(f"Empty ROI for frame {frame}, box {i}")
                    continue


                #Calculate the average flow vector in this region
                mean_flow = np.mean(roi_flow, axis=(0, 1))

                #Calculate expected motion based on bounding box movement (from annotations)
                if frame < total_frames - 1:
                    nannotations = self.get_annotation(frame + 1)
                    nx1, ny1, nx2, ny2 = nannotations[i].x1, nannotations[i].y1, nannotations[i].x2, nannotations[i].y2
                    if nannotations[i]:
                        expected_dx = (nx1 + nx2) / 2 - (x1 + x2) / 2
                        expected_dy = (ny1 + ny2) / 2 - (y1 + y2) / 2

                        #Displacement error between predicted flow and actual movement
                        displacement_error = np.sqrt((mean_flow[0] - expected_dx)**2 + (mean_flow[1] - expected_dy)**2)
                        total_displacement_error += displacement_error

        average_displacement_error = total_displacement_error / total_frames
        return average_displacement_error
    
    def evaluate_sparse_optical_flow(self, flow_sequence):
        total_displacement_error = 0
        total_points = 0

        for frame in range(len(flow_sequence) - 1):
            annotations = self.get_annotation(frame)
            predicted_points = flow_sequence[frame]
            next_predicted_points = flow_sequence[frame + 1]
            current_annotations = self.get_annotation(frame)
            next_annotations = self.get_annotation(frame + 1)

            for i in range(len(predicted_points)):
                pred_x, pred_y = predicted_points[i]
                next_pred_x, next_pred_y = next_predicted_points[i]

                # Get the corresponding ground truth points
                gt_x1, gt_y1, gt_x2, gt_y2 = current_annotations[i].x1, current_annotations[i].y1, current_annotations[i].x2, current_annotations[i].y2
                next_gt_x1, next_gt_y1, next_gt_x2, next_gt_y2 = next_annotations[i].x1, next_annotations[i].y1, next_annotations[i].x2, next_annotations[i].y2

                # Calculate the ground truth center points
                gt_center_x = (gt_x1 + gt_x2) / 2
                gt_center_y = (gt_y1 + gt_y2) / 2
                next_gt_center_x = (next_gt_x1 + next_gt_x2) / 2
                next_gt_center_y = (next_gt_y1 + next_gt_y2) / 2

                # Calculate expected motion based on bounding box movement
                expected_dx = next_gt_center_x - gt_center_x
                expected_dy = next_gt_center_y - gt_center_y

                # Displacement error between predicted flow and actual movement
                displacement_error = np.sqrt((next_pred_x - (pred_x + expected_dx))**2 + (next_pred_y - (pred_y + expected_dy))**2)
                total_displacement_error += displacement_error
                total_points += 1

        average_displacement_error = total_displacement_error / total_points if total_points > 0 else float('inf')
        return average_displacement_error

    def evaluate_tracking2(self, results, boxes, order, IoU_threshold = 0.2):

        n_frames = []
        
        tps = []
        fps = []
        fns = []
        id_switches = []
        n_annotations = []
        
        ids = []
        for i in range(len(results[0])):
            ids.append(i) # index and value are the same


        for i in range(len(results)):

            frame_ann = self.get_annotation(i)

            frame_tp = 0
            frame_fp = 0
            frame_fn = 0

            frame_n_annotations = 0
            frame_id_switches = 0

            matched_ious = []
            n_matches = 0
            

            for j in range(len(results[i])):

                pred_result = results[i][j]
                pred_box = [boxes[i][j][0], boxes[i][j][1], boxes[i][j][2] + boxes[i][j][0], boxes[i][j][3] + boxes[i][j][1]]
                ann_box = [frame_ann[j].x1, frame_ann[j].y1, frame_ann[j].x2, frame_ann[j].y2]

                if ann_box == [0, 0, 0, 0]: # player annotated as not in frame
                    frame_n_annotations += 1
                    if pred_result:
                        frame_fp += 1
                    else:
                        frame_tp += 1
                else: # player annotated in frame
                    frame_n_annotations += 1
                    if not pred_result:
                        frame_fn += 1 # tracker did not find the player
                    else:
                        iou = self.calculate_iou(pred_box, ann_box)
                        if iou > IoU_threshold: # tracker found the correct player
                            frame_tp += 1
                            matched_ious.append(iou)
                            n_matches += 1
                            if j != ids[j]: # the player found id is different than the last one
                                frame_id_switches += 1
                            ids[j] = j
                        else:
                            for k in range(len(frame_ann)):
                                other_box = [frame_ann[k].x1, frame_ann[k].y1, frame_ann[k].x2, frame_ann[k].y2]
                                iou = self.calculate_iou(pred_box, other_box)
                                if iou > IoU_threshold: # tracker found another player instead of the correct one
                                    frame_fp += 1
                                    if k != ids[j]: # the player found id is different than the last one
                                        frame_id_switches += 1
                                    ids[j] = k
                                    break
            
            n_frames.append(i)
            tps.append(frame_tp)
            fps.append(frame_fp)
            fns.append(frame_fn)
            n_annotations.append(frame_n_annotations)
            id_switches.append(frame_id_switches)
        

        precisions = []
        recalls = []
        f1s = []
        for i in range(len(n_frames)):
            try:
                precisions.append(tps[i] / (tps[i] + fps[i]))
            except ZeroDivisionError:
                precisions.append(0)
            try:
                recalls.append(tps[i] / (tps[i] + fns[i]))
            except ZeroDivisionError:
                recalls.append(0)
            try:
                f1s.append(2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]))
            except ZeroDivisionError:
                f1s.append(0)
        
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1 = np.mean(f1s)
        total_id_switches = sum(id_switches)
        print(f"Sum of false positives: {sum(fps)}")
        print(f"Sum of false negatives: {sum(fns)}")
        mota = 1 - (sum(fps) + sum(fns) + total_id_switches) / sum(n_annotations)
        motp = sum(matched_ious) / n_matches if n_matches > 0 else 0

        return mean_precision, mean_recall, mean_f1, total_id_switches, mota, motp