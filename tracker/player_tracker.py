from ultralytics import YOLO
import cv2
import pickle
import sys
import numpy as np
sys.path.append('../')
from utils import get_minimum_distance, find_middle

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def __str__(self):
        return str(self.model)
    
    def detect_frame(self, frame):
        result = self.model.track(frame, persist=True)[0]   
        id_class_dict = result.names
        
        player_dict = {}
        for box in result.boxes:
            track_id = int(box.id.tolist()[0])
            box_result = box.xyxy.tolist()[0]
            box_class_id = box.cls.tolist()[0]
            box_class_name = id_class_dict[box_class_id]
            if box_class_name == "person":
                player_dict[track_id] = box_result
        
        return player_dict
    
    def detect_frames(self,frames,read_from_stub = False, stub_path = None):
        player_detections = []

        if read_from_stub:
            with open(stub_path,"rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_detections.append(self.detect_frame(frame))

        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(player_detections,f)
        return player_detections
    def find_player(self,court_keypoints, player_detection):
        player_dict = player_detection
        players = []
        for id, bbox in player_dict.items():
            min_distance = np.inf
            player_point = find_middle(bbox)
            distance = get_minimum_distance(player_point,court_keypoints)
            players.append((id,distance))
        players.sort(key=lambda x: x[1])
        return (players[0][0],players[1][0])

    def filters_players_only(self, court_keypoints,player_detections):
        new_player_detections = []
        selected_track_ids  = self.find_player(court_keypoints,player_detections[0])
        for player_dict in player_detections:
            player_detect = {id: bbox for id,bbox in player_dict.items() if id in selected_track_ids}
            new_player_detections.append(player_detect)

        return new_player_detections

    
    def draw_bboxes(self,frames,player_detections):
        output_frames = []
        for frame, player_dict in zip(frames,player_detections):
            for track_id, box_result in player_dict.items():
                x1, y1, x2, y2 = box_result
                cv2.putText(frame,text=f"Player ID: {track_id}",
                            org=(int(x1),int(y1)-10),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            color = (0,0,255),
                            fontScale=1,
                            thickness=2)
                cv2.rectangle(img = frame, 
                                     pt1= (int(x1),int(y1)), pt2 = (int(x2),int(y2)),
                                     color=(0,0,255),
                                     thickness=2)
            output_frames.append(frame)
        return output_frames