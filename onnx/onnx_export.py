#!/usr/bin/env python

import torch
from os import makedirs
from os.path import dirname
from PIL import Image
import torch.nn as nn
from load import load_model, transform

DEVICE = torch.device("cpu")


class Eta(nn.Module):

  def __init__(self):
    super(Eta, self).__init__()
    self.model = MODEL

  def forward(self, img):
    with torch.no_grad():
      img = img.to(DEVICE)
      result, _, _ = self.model(img)
      result = result.squeeze()
      # print(result)
      return result


opset_version = 17

onnx_fp = 'onnx/eta/tad66k.onnx'
makedirs(dirname(onnx_fp))


def onnx_export(model, args, **kwds):

  torch.onnx.export(
      model,
      args,
      onnx_fp,
      export_params=True,
      # verbose=True,
      opset_version=opset_version,
      do_constant_folding=False,
      output_names=['output'],
      **kwds)


MODEL = load_model("../AVA/AVA_AOT_vacc_0.8259_srcc_0.7596_vlcc_0.7710.pth")
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

# onnx_model = onnx.load(onnx_fp)  # load onnx model
# model_simp, check = simplify(onnx_model)
# print(check)
# onnx.save(model_simp, join(dirname(onnx_fp), 'simplify.' + basename(onnx_fp)))
# print(onnx_fp, "DONE\n")
