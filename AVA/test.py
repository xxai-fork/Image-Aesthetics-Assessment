#!/usr/bin/env python

import torch
from PIL import Image
from models import build_model
from config import get_config

img = Image.open('1.jpg').resize((224, 224))
ck = torch.load('./AVA_AOT_vacc_0.8259_srcc_0.7596_vlcc_0.7710.pth',
                map_location=torch.device('mps'))
model = build_model(get_config())
model.load_state_dict(ck)
print(ck['model'])
# model.eval()
# device = torch.device("mps")
# model.to(device)
#
# with torch.no_grad():
#   pred, _, _ = model(img)
#
# print(pred)
