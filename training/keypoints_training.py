import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms, models
import json
import os
import yaml
import numpy as np
import cv2
import sys
sys.path.append('../')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('training/config.yml','r') as f:
   config = yaml.load(f,Loader=yaml.SafeLoader)


class KeyPointsDataset(Dataset):
  def __init__(self,data_dir,train=False,val=False):
    self.image_dir = os.path.join(data_dir,'images')
    if train:
      self.json_file = os.path.join(data_dir,'data_train.json')
    if val:
      self.json_file = os.path.join(data_dir,'data_val.json')
    with open(self.json_file,'r') as f:
      self.data = json.load(f)
    self.transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    item = self.data[idx]
    imagefile = item['id'] + '.png'
    img = cv2.imread(os.path.join(self.image_dir,imagefile))
    h,w = img.shape[0], img.shape[1]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = self.transforms(img)
    kps = np.array(item['kps']).astype(np.float32)
    kps = kps.flatten()

    kps[::2] *= 224.0/w
    kps[1::2] *= 224.0/h

    return img,kps

class TrainKeypoints:
  def __init__(self,model,data_dir):
      self.model = model
      self.DATA_DIR = data_dir
      trainset = KeyPointsDataset(self.DATA_DIR,train=True)
      valset = KeyPointsDataset(self.DATA_DIR,val=True)
      self.trainloader = DataLoader(trainset,batch_size=config['court']['BATCH_SIZE'],shuffle=True)
      self.valloader = DataLoader(valset,batch_size=config['court']['BATCH_SIZE'],shuffle = True)
      self.EPOCHS = config['court']['EPOCHS']
      self.criterion = nn.MSELoss()
      self.optimizer = torch.optim.Adam(model.parameters(),lr=config['court']['learning_rate'])
  def build_model(self):
      model = models.resnet34(pretrained = True)
      self.model.fc = nn.Linear(model.fc.in_features, out_features = 14*2)
      self.model.to(DEVICE)

  def train(self):
      self.model.train()
      for epoch in range(self.EPOCHS):
          total_loss = 0.0
          for i,(img,kps) in enumerate(self.trainloader):
              img = img.to(DEVICE)
              kps = kps.to(DEVICE)
              self.optimizer.zero_grad()
              outputs = self.model(img)
              loss = self.criterion(outputs,kps)
              loss.backward()
              self.optimizer.step()
              total_loss += loss.item() * img.size(0)
          epoch_loss = total_loss/len(self.trainloader)
          print(f'Epoch {epoch+1}/{self.EPOCHS}, loss: {round(epoch_loss,3)}')
      torch.save(self.model.state_dict(), 'keypoints.pth')
