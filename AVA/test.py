#!/usr/bin/env python

import torch
from PIL import Image
from models.dat import DAT
import yaml
from torchvision import transforms
import numpy as np


def get_score(y_pred):
  w = torch.from_numpy(np.linspace(1, 10, 10))
  w = w.type(torch.FloatTensor)
  w = w.to(device)

  w_batch = w.repeat(y_pred.size(0), 1)

  score = (y_pred * w_batch).sum(dim=1)
  score_np = score.data.cpu().numpy()
  return score_np[0]


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
transform = transforms.Compose([transforms.ToTensor(), normalize])

ck = torch.load('./AVA_AOT_vacc_0.8259_srcc_0.7596_vlcc_0.7710.pth',
                map_location=torch.device('mps'))

with open('./configs/dat_base.yaml') as f:
  conf = yaml.load(f, Loader=yaml.FullLoader)['MODEL']['DAT']

model = DAT(**conf)
model.load_state_dict(ck)
model.eval()
device = torch.device("cpu")
model.to(device)

for i in range(1, 7):
  img = Image.open(f'jpg/{i}.jpg').resize((224, 224))
  img = transform(img)
  # 参数是一个图片的数组， unsqueeze相当于创建一个只有一个图片的数组
  img = img.unsqueeze(0)
  img = img.to(device)

  with torch.no_grad():
    pred, _, _ = model(img)

  pred = get_score(pred)
  print(i, pred)
