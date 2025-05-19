#!/usr/bin/bash

module load stack/2024-06
module load gcc/12.2.0
source source /cluster/home/dmuehlema/LoRA-ViT-Ensemble/venv/bin/activate
module load python_cuda/3.11.6

#python main.py SST2_settings_LoRA1.json LoRABert 2

/cluster/home/dmuehlema/LoRA-ViT-Ensemble/venv/bin/python main.py $1 $2 $3
#if [ -n "$4" ]; then
#    /cluster/home/dmuehlema/LoRA-ViT-Ensemble/venv/bin/torchrun --nproc_per_node=4 main_multiGPU_ckpt.py $1 $2 $3 $4
#else
#    /cluster/home/dmuehlema/LoRA-ViT-Ensemble/venv/bin/torchrun --nproc_per_node=4 main_multiGPU_ckpt.py $1 $2 $3
#fi