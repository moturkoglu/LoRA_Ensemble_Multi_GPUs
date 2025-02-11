#!/usr/bin/env python

"""
implements the training and testing pipeline for CIFAR10
"""

### IMPORTS ###
# Built-in imports
import sys

# Lib imports

# Custom imports
from experiment_settings.settings_multiGPU import get_settings
from model_training_evaluation.train_evaluate_multiGPU import train_evaluate_ensemble
import const
import utils
from models.model_loader import load_model
import json
import pandas as pd
from model_training_evaluation.cross_validation import cross_validation

### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### EXECUTE ###
if __name__ == "__main__":
    """
    Function for training and/or evaluating ViT ensemble models.
    """
    
    if len(sys.argv) > 1:
        # Iterate over the command-line arguments
        print("Command-line arguments:")
        print("1. Name of the json file containing the experiment settings: ", sys.argv[1])
        print("2. Type of the ensemble model: ", sys.argv[2])
        print("3. Number of the ensemble members: ", sys.argv[3])
        print("")

        # Set the path to the experiment settings
        settings_path = const.SETTINGS_DIR.joinpath(sys.argv[1])

        # Get the experiment settings
        settings = get_settings(path = settings_path, ensemble_type= sys.argv[2], nr_members = int(sys.argv[3]))

        # If a saved model is provided, load the model
        if len(sys.argv) == 5:
            settings["model_settings"]["model"] = load_model(sys.argv[4], settings["model_settings"]["model"])
            settings["training_settings"]["training"] = False

        # If cross-validation is enabled, perform cross-validation
        if "cross_validation" in settings["training_settings"].keys() and settings["training_settings"]["cross_validation"] == True:
            cross_validation(settings, sys.argv[1])

        # If training is enabled, train/evaluate the model
        else:
            # Train and evaluate the ensemble model
            train_evaluate_ensemble(settings)

    else:
        print("Not all command-line arguments are provided.")
        print("Please provide the following command-line arguments:")
        print("1. Name of the json file containing the experiment settings.")
        print("2. Type of the ensemble model ('Deep-Ensemble' , 'LoRA-Former').")
        print("3. Number of the ensemble members.")
        print("")
        print("Example: python main.py CIFAR10_settings_experiment1.json LoRA-Former 2")

        # Running multi-GPU on cluster using shell scripts (Linux)
        # 1. Change variable "cluster" in settings_multiGPU.py to True if not already
        # 2. Change variable "cluster" in const.py to True if not already
        # 3. Change path for saving model on cluster in const.py, otherwise models are saved in scratch
        # 4. Change --nproc_per_node=4 to the number of GPUs to use in the file batch_script.slurm
        # 5. Change time=48 and mem=20000 to the desired time and memory in the file run_slurm_jobs.sh for the desired number of members
        # 6. Change --gpus=4 in the file run_slurm_jobs.sh to the number of GPUs to use
        # bash run_slurm_jobs.sh -t 'Deep_Ensemble' -s 'INat2017_settings_explicitlargerLR' -e '1,2' -n '1,8'
        # where -t 'Deep_Ensemble' is the type of ensemble model
        # where -s 'INat2017_settings_explicitlargerLR' is the name of the json file containing the experiment settings without the experiment number
        # where -e '1,2' is the number of experiments to run
        # where -n '1,8' is the number of ensemble members to train
        # the example above will start the settings INat2017_settings_explicitlargerLR1.json and INat2017_settings_explicitlargerLR2.json with 
        # 1 and 8 ensemble members respectively. In total 4 jobs will be started

        # Running multi-GPU on euler (Linux)
        # 1. Change variable "cluster" in settings_multiGPU.py to True if not already
        # 2. Change variable "cluster" in const.py to True if not already
        # 3. Change path for saving model on cluster in const.py, otherwise models are saved in scratch
        # torchrun --nproc_per_node=4 --master_port=12355 main_multiGPU.py CIFAR10/final/CIFAR10_settings_explicitScores1.json Deep_Ensemble 4 
        # where --nproc_per_node=4 is the number of GPUs to use

        # Running multi-GPU local on Windows
        # 1. Change variable "cluster" in settings_multiGPU.py to False
        # 2. Change variable "cluster" in const.py to False
        # 3. Change local path to INat dataset if needed in settings_multiGPU.py
        # python -m torch.distributed.run --nproc_per_node=1 main_multiGPU.py INat2017_settings_ExplicitmultiGPUtest1.json Deep_Ensemble 1
        # where --nproc_per_node=1 is the number of GPUs to use



