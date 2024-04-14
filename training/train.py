from ball_detection_training import BallDetectionTraining
from keypoints_training import TrainKeypoints
import gdown
from zipfile import ZipFile 
import sys
sys.path.append('../')

def main():
    print("Tennis Court dataset to be downloaded...")
    file_id = '1lhAaeQCmk2y440PmagA0KmIVBIysVMwu'
    output = "data/tennis_court.zip"
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
    with ZipFile("data/tennis_court.zip", 'r') as zObject:  
        zObject.extractall(path="data/tennis_court")
    print("Tennis Court dataset downloaded")

    print("Tennis Ball dataset to be downloaded...")
    api_key  = input("Enter your Roboflow API Key: ")
    balldetect = BallDetectionTraining('data/data.yaml',api_key )
    balldetect.train()

    court_train = TrainKeypoints()
    court_train.build_model()
    court_train.train()
    print("Training Complete")

if __name__ == "__main__":
    main()