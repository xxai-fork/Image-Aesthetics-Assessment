#!/usr/bin/env python

import torch
from models.dat import DAT
import yaml
from torchvision import transforms

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
