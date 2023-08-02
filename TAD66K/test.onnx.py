#!/usr/bin/env python

import onnxruntime

# Create a session options object
options = onnxruntime.SessionOptions()

# Retrieve the available providers
providers = onnxruntime.get_available_providers()

# Print the list of available providers
print(f"Available ONNX Runtime providers: {providers}")

session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2


def onnx_load(fp):
  sess = onnxruntime.InferenceSession(fp,
                                      sess_options=session,
                                      providers=providers)
  return sess


sess = onnx_load('./onnx/eta/tad66k.onnx')

if __name__ == "__main__":
  # main()
  pass
