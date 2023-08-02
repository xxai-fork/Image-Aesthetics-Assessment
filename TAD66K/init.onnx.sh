#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

pip uninstall onnxruntime
pip install onnxruntime-silicon
