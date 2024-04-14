
# Tennis Match Tracker

This project utilizes OpenCV, YOLO and CNN to track the position, movement of players in a video. YOLOv8 is used to track the players. YOLOv5 is used to track the position of tennis ball at every frame of the video. ResNet34 is fine-tuned to detect the court keypoints. 

# Weights

Download the pretrained weights

https://drive.google.com/file/d/11oPP9h-lVfuOv09RFt0mmvydy4-maTjV/view?usp=sharing, 

https://drive.google.com/file/d/1iGQMabajjVm_MbUlCXJDnslnLmYvgleX/view?usp=sharing, 

https://drive.google.com/file/d/1ihueDeTl2XiYDiVMYBKpGygnBMG91pIW/view?usp=sharing


## Screenshots

![Demo output](https://github.com/abhroroy365/Tennis-Tracker/blob/master/output/output_video.gif)


## Run Locally

Clone the project

```bash
  git clone https://github.com/abhroroy365/Tennis-Tracker.git
```

Go to the project directory

```bash
  cd Tennis-Tracker
```

Create virtual environment

```bash
  python -m venv env
```
Activate the virtual environment

```bash
  env\Scripts\activate
```

Install dependencies
```bash
  pip install -r requirements.txt
```

For training (no need, weights alrady provided)

```bash
  python .\training\train.py
```

For running the tracker on your video
```bash
  python run-tracker.py
```

## 🛠 Skills
Pytorch, OpenCV, YOLO, Python, Computer Vision

