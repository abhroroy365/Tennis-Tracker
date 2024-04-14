import torch
from torchvision import transforms
import cv2
import numpy as np
from torchvision import models
import torch.nn as nn

class CourtLineDetector:
    def __init__(self,model_path):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet34(pretrained = True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load( model_path, map_location = self.DEVICE))

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img_tensor = self.transforms(img).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            predictions = self.model(img_tensor)
        keypoints = predictions.squeeze().cpu().numpy()

        original_h, original_w= image.shape[:2]

        keypoints[::2] *= original_w /224.0
        keypoints[1::2] *= original_h /224.0

        return keypoints
            
    def draw_points(self,frame,keypoints):
        for i in range(0,len(keypoints),2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(frame,
                        text = str(i//2),
                        org = (x,y-10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(255,0,0),
                        thickness=2
                        )
            cv2.circle(frame,
                       center = (x,y),
                       radius= 5,
                       color = (0,255,0),
                       thickness=1)
        return frame
    
    def draw_points_video(self, video_frames,keypoints):
        output_frames = []
        for frame in video_frames:
            output_frames.append(self.draw_points(frame,keypoints))
        return output_frames
            