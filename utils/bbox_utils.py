import numpy as np

def find_middle(bbox):
    x1,y1,x2,y2 = bbox
    x = (x1+x2)/2
    y = (y1+y2)/2
    return (x,y)

def calculate_distance(court_point,player_middle):
    distance = ( (court_point[0]-player_middle[0])**2 + (court_point[1]-player_middle[1])**2)**0.5
    return distance

def get_minimum_distance(player_middle,court_keypoints):
    min_distance  = np.inf
    for i in range(0,len(court_keypoints),2):
        x = int(court_keypoints[i])
        y = int(court_keypoints[i+1])
        distance = calculate_distance((x,y), player_middle)
        if distance < min_distance:
            min_distance = distance
    
    return min_distance

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   closest_distance = float('inf')
   key_point_ind = keypoint_indices[0]
   for keypoint_indix in keypoint_indices:
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
       distance = abs(point[1]-keypoint[1])

       if distance<closest_distance:
           closest_distance = distance
           key_point_ind = keypoint_indix
    
   return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))