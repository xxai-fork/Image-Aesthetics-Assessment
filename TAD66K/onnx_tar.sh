#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

tar -jcvf onnx/eta.tar.bz2 -C onnx eta.tad66k.onnx
