#!/usr/bin/env bash
#TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5" bash compile.sh
PYTHON=${PYTHON:-"python"}

echo "Building box_iou_rotated..."
cd box_iou_rotated
#if [ -d "build" ]; then
#    rm -r build
#fi
$PYTHON setup.py build_ext --inplace

cd ../convex
#if [ -d "build" ]; then
#    rm -r build
#fi
$PYTHON setup.py build_ext --inplace