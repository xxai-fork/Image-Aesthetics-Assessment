#!/usr/bin/env python

import onnxruntime
from test import diff

options = onnxruntime.SessionOptions()

providers = onnxruntime.get_available_providers()[1:]

print(providers)


def onnx_load(fp):
  session = onnxruntime.SessionOptions()

  option = onnxruntime.RunOptions()
  option.log_severity_level = 2
  sess = onnxruntime.InferenceSession(fp,
                                      sess_options=session,
                                      providers=providers)
  return sess


sess = onnx_load('./onnx/eta.tad66k.onnx')
# sess = onnx_load('./onnx/onnxoptimizer.eta.tad66k.onnx')


def run(img):
  return sess.run(None, {'input': img})[0]


def to_numpy(i):
  return i.numpy()


diff(run, to_numpy)

if __name__ == "__main__":
  # main()
  pass
