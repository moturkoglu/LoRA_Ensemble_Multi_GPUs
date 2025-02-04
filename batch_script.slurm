#!/usr/bin/bash

module load stack/2024-06
module load gcc/12.2.0
source venv/bin/activate
module load python_cuda/3.11.6

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12355 main_multiGPU.py $1 $2 $3