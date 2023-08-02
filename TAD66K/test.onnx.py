#!/usr/bin/env python

import onnxruntime

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
# sess = onnx_load('./onnx/eta/onnxoptimizer.tad66k.onnx')
print(sess.get_inputs())

if __name__ == "__main__":
  # main()
  pass
