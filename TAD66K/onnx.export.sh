#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

rm -rf ./onnx
./onnx_export.py
cd onnx/eta
onnx=tad66k.onnx
mod=onnxoptimizer
python -m $mod $onnx $mod.$onnx
# mv $mod.$onnx $onnx
