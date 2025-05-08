#!/bin/bash
# run.sh

export CUDA_AVAILAVLE_DEVICE=0
python src/analyze_over_smoothing/analyze_over_smoothing.py --residual_type mix --config src/analyze_over_smoothing/config.yaml