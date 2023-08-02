#!/usr/bin/env python

import os
import torch
from numpy import mean
from os.path import join
from PIL import Image
from models.dat import DAT
import yaml
from torchvision import transforms
import numpy as np
from fire import Fire


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

device = torch.device("cpu")


def load_model(model_name):
  ck = torch.load(model_name, map_location=torch.device('mps'))

  with open('./configs/dat_base.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)['MODEL']['DAT']

  model = DAT(**conf)
  model.load_state_dict(ck)
  model.eval()
  model.to(device)
  return model


def jpg_iter(root):
  for root, dirs, files in os.walk(join('../jpg', root)):
    # 遍历当前目录下所有文件
    for filename in files:
      # 检查文件名是否以'.jpg'结尾
      if filename.endswith('.jpg'):
        # 拼接完整路径
        jpg_path = os.path.join(root, filename)
        yield jpg_path


def score_dir(root, model):
  r = []
  for i in jpg_iter(root):
    img = Image.open(i).resize((224, 224))
    img = transform(img)
    # 参数是一个图片的数组， unsqueeze相当于创建一个只有一个图片的数组
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
      pred, _, _ = model(img)

    s = get_score(pred)
    print(i, s)
    r.append(s)
  return r


def li_normalize(arr):
  arr_min = np.min(arr)
  arr_max = np.max(arr)
  return (arr - arr_min) / (arr_max - arr_min)


@Fire
def main(model_name):
  model = load_model(model_name)
  good = score_dir('good', model)
  bad = score_dir('bad', model)
  li = li_normalize(good + bad)

  len_good = len(good)
  diff = mean(li[0:len_good]) / mean(li[len_good:])
  print(model_name + '\n' + 'DIFF %.2f%%' % (100 * diff))
