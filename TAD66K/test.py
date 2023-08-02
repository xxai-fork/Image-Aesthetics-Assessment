#!/usr/bin/env python

from load import load_model, transform, device, normalize
import os
import numpy as np
from numpy import mean
from PIL import Image
import torch
from os.path import abspath, dirname, join

ROOT = dirname(dirname(abspath(__file__)))


def jpg_iter(root):
  for root, dirs, files in os.walk(join(ROOT, 'jpg', root)):
    # 遍历当前目录下所有文件
    for filename in files:
      # 检查文件名是否以'.jpg'结尾
      if filename.endswith('.jpg'):
        # 拼接完整路径
        jpg_path = os.path.join(root, filename)
        yield jpg_path


def run(model, img):
  with torch.no_grad():
    pred, _, _ = model(img)
  pred = pred.data.cpu().numpy()[0][0]
  return pred


def score_dir(root, model, parse):
  r = []
  for i in jpg_iter(root):
    img = Image.open(i)
    img = img.resize((224, 224))
    # 参数是一个图片的数组， unsqueeze相当于创建一个只有一个图片的数组
    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    img = parse(img)

    pred = model(img)
    print(i[len(ROOT) + 5:], pred)
    r.append(pred)
  return r


def li_normalize(arr):
  arr_min = np.min(arr)
  arr_max = np.max(arr)
  return (arr - arr_min) / (arr_max - arr_min)


def empty(i):
  return i


def main(model_name):
  model = load_model(model_name)

  def _run(img):
    return run(model, img)

  for parse in [normalize, empty]:
    print('#', parse)

    bad = score_dir('bad', _run, parse)
    good = score_dir('good', _run, parse)
    li = li_normalize(good + bad)

    len_good = len(good)
    diff = mean(li[0:len_good]) / mean(li[len_good:])
    print(model_name + '\n' + '好图平均分 / 差图平均分  %.2f%%' % (100 * diff))


if __name__ == "__main__":
  main('./TAD66K_AOT_vacc_0.6882_srcc_0.5171_vlcc_0.5460.pth')
