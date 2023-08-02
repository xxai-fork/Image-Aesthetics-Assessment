#!/usr/bin/env python

import onnxruntime
from test import score_dir

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


sess = onnx_load('./onnx/eta/tad66k.onnx')


def run(img):
  sess.run(None, {'input': img})


score_dir('../jpg', run)

if __name__ == "__main__":
  # main()
  pass
