from utils import (read_video, 
                   save_video,
                   calculate_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
from tracker import PlayerTracker,BallTracker,CourtLineDetector
import os
import cv2
import pandas as pd
from mini_court import MiniCourt 
from copy import deepcopy
import constants

def main():
    # Read video
    input_path = 'data/test_input_video.mp4'
    video_frames = read_video(input_path)
    cwd = os.getcwd()
    # detect court keypoints
    court_model_path = os.path.join(cwd,'saved_models\KEYPOINTSMODEL\keypoints2.pth')
    court_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_detector.predict(video_frames[0])

    # track player
    player_tracker_stub = '.\saved_models\PLAYER_TRACK'  # stub path to save the player tracker
    if len(os.listdir(player_tracker_stub)) == 1:  # check if there is already a tracker
        read_from_stub = False
    else:
        read_from_stub = True 
    player_tracker_stub = os.path.join(player_tracker_stub,'player_tracker.pkl')
    player_model_path = os.path.join(cwd,'saved_models\PLAYER_TRACK\yolov8x.pt')
    player_tracker = PlayerTracker(player_model_path)
    player_detections = player_tracker.detect_frames(video_frames,read_from_stub,player_tracker_stub)
    player_detections = player_tracker.filters_players_only(court_keypoints,player_detections)

    # track ball
    ball_tracker_stub = '.\saved_models\BALL_TRACK'  # stub path to save the player tracker
    if len(os.listdir(ball_tracker_stub)) == 1:  # check if there is already a tracker
        read_from_stub = False
    else:
        read_from_stub = True 
    ball_tracker_stub = os.path.join(ball_tracker_stub,'ball_tracker.pkl')
    ball_model_path = os.path.join(cwd,'saved_models\BALL_TRACK\yolo5_last.pt')
    ball_tracker = BallTracker(ball_model_path)
    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub,ball_tracker_stub)
    ball_detections = ball_tracker.interpolate_missing_ball_positions(ball_detections)

    # MiniCourt
    minicourt = MiniCourt(video_frames[0])
    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = minicourt.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = calculate_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           minicourt.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: calculate_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = calculate_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           minicourt.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    # draw boxes
    video_frames_bbox1 = court_detector.draw_points_video(video_frames,court_keypoints) # court
    video_frames_bbox2 = ball_tracker.draw_bboxes(video_frames_bbox1,ball_detections)  # ball
    video_frames_bbox3 = player_tracker.draw_bboxes(video_frames_bbox2,player_detections)  # players

    # draw minicourt
    video_frames_bbox4 = minicourt.draw_mini_court(video_frames_bbox3)
    video_frames_bbox5 = minicourt.draw_points_on_mini_court(video_frames_bbox4,player_mini_court_detections)
    video_frames_bbox6 = minicourt.draw_points_on_mini_court(video_frames_bbox5,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    video_frames_bbox7 = draw_player_stats(video_frames_bbox6,player_stats_data_df)
    # add frame number to every frame
    for i, frame in enumerate(video_frames_bbox7):
        cv2.putText(frame,
                    text= f'Frame No. {i+1}',
                    org=(10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color = (0,255,0),
                    thickness=2 
                    )
    save_video(video_frames_bbox4,'output\output_video.avi')

if __name__ == '__main__':
    main()