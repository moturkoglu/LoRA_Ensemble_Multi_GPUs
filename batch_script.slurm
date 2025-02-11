#!/usr/bin/bash

module load stack/2024-06
module load gcc/12.2.0
source source /cluster/home/dmuehlema/LoRA-ViT-Ensemble/venv/bin/activate
module load python_cuda/3.11.6

/cluster/home/dmuehlema/LoRA-ViT-Ensemble/venv/bin/torchrun --nproc_per_node=4 main_multiGPU.py $1 $2 $3  