from ultralytics import YOLO
from roboflow import Roboflow
import torch
import sys
import yaml
sys.path.append('../')

with open('config.yml','r')as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
class BallDetectionTraining:
    def __init__(self,data_yaml,api):
        self.rf = Roboflow(api_key=api)
        project = self.rf.workspace("tennisobjectdetect").project("tennis-ybzmd")
        version = project.version(6)
        dataset = version.download("yolov5")
        self.model = YOLO("./data/yolov8x.yaml")
        self.model = YOLO(config['ball']['model'])
        self.data = data_yaml
    def train(self):
        self.model.train(data=self.data, epochs=config['ball']['epochs'],imgsz=640)
        torch.save(self.model, '..\saved_models\BALL_TRACK\yolo5_last.pt')
        print("Model saved")
