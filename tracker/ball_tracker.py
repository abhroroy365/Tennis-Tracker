from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def __str__(self):
        return str(self.model)
    
    def detect_frame(self, frame):
        result = self.model.predict(frame)[0]   
        ball_dict = {}
        for box in result.boxes:
            box_result = box.xyxy.tolist()[0]
            ball_dict[1] = box_result
        
        return ball_dict
    
    def detect_frames(self,frames,read_from_stub = False, stub_path = None):
        ball_detections = []

        if read_from_stub:
            with open(stub_path,"rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_detections.append(self.detect_frame(frame))

        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(ball_detections,f)
        return ball_detections
    
    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits
    
    def interpolate_missing_ball_positions(self,ball_positions):
        position_list = [x.get(1,[]) for x in ball_positions]
        position_df = pd.DataFrame(position_list, columns=['x1','y1','x2','y2'])
        position_df = position_df.interpolate()
        position_df = position_df.bfill()
        ball_positions = [{1:x} for x in position_df.to_numpy().tolist()]
        return ball_positions
    
    def draw_bboxes(self,frames,ball_detections):
        output_frames = []
        for frame, ball_dict in zip(frames,ball_detections):
            for i, box_result in ball_dict.items():
                x1, y1, x2, y2 = box_result
                cv2.rectangle(img = frame, 
                                     pt1= (int(x1),int(y1)), pt2 = (int(x2),int(y2)),
                                     color=(0,0,255),
                                     thickness=2)
            output_frames.append(frame)
        return output_frames