#!/usr/bin/env python

import torch
from os import makedirs
from os.path import dirname
from PIL import Image
import torch.nn as nn
from load import load_model, transform, normalize

DEVICE = torch.device("cpu")


class Eta(nn.Module):

  def __init__(self):
    super(Eta, self).__init__()
    self.model = MODEL

  def forward(self, img):
    with torch.no_grad():
      img = normalize(img)
      img = img.to(DEVICE)
      result, _, _ = self.model(img)
      result = result.squeeze()
      # print(result)
      return result


opset_version = 18


def onnx_export(model, args, **kwds):
  fp = 'onnx/eta/tad66k.onnx'
  makedirs(dirname(fp))

  torch.onnx.export(
      model,
      args,
      fp,
      export_params=True,
      # verbose=True,
      opset_version=opset_version,
      do_constant_folding=False,
      output_names=['output'],
      **kwds)
  print(fp, "DONE\n")


MODEL = load_model("TAD66K_AOT_vacc_0.6882_srcc_0.5171_vlcc_0.5460.pth")
MODEL.to(DEVICE)

img = transform(Image.open('../jpg/good/1.jpg').resize((224, 224)))
img = img.unsqueeze(0).to(DEVICE)

ETA = Eta()

# print(ETA.forward(img))
onnx_export(ETA,
            img,
            input_names=['input'],
            dynamic_axes={'input': {
                0: 'batch'
            }})