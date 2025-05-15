#!/usr/bin/env python

"""
Implements the training pipeline for ViT and BERT (SST2) ensembles.
"""

### IMPORTS ###
import time
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray

from utils_uncertainty import _ECELoss, function_space_analysis
from models.lora_ensemble import BatchMode
from models.model_loader import load_model
from utils_GPU import DEVICE
import const

### METADATA ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.3"


def train_evaluate_ensemble(settings: dict, batch_mode: BatchMode = BatchMode.DEFAULT) -> None:
    """
    Train and evaluate ViT or BERT (SST2) ensemble models.
    """
    model = settings["model_settings"]["model"]
    n_members = settings["model_settings"]["nr_members"]

    # Move loss weights to device
    settings["training_settings"]["loss"].weight = settings["training_settings"]["loss"].weight.to(DEVICE)
    optimizer = settings["training_settings"]["optimizer"]
    lr_schedule = settings["training_settings"]["lr_schedule"]
    criterion = settings["training_settings"]["loss"]

    if settings["training_settings"]["training"]:
        train_loader = settings["training_settings"]["training_dataloader"]
    if settings["evaluation_settings"]["evaluation"]:
        val_loader = settings["evaluation_settings"]["evaluation_dataloader"]

    scaler = torch.cuda.amp.GradScaler(enabled=settings["training_settings"]["use_amp"])
    gradient_updates = 0
    finish_training = False

    for epoch in range(settings["training_settings"]["max_epochs"]):
        model.train()
        train_loss = 0.0

        if settings["training_settings"]["training"]:
            with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    
                    # # optional subset
                    # max_iter = 1
                    # if max_iter and batch_idx >= max_iter:
                    #     break

                    optimizer.zero_grad()

                    # Forward pass
                    if settings["data_settings"]["data_set"] == "SST2":
                        # Batch is dict with input_ids, attention_mask, labels
                        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                            enabled=settings["training_settings"]["use_amp"]):
                            outputs = model(**{k: inputs[k] for k in inputs if k != 'labels'})
                        # Extract logits
                        # logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        logits = outputs
                        labels = inputs['labels']

                    else:
                        data_train, labels = batch
                        data_train = data_train.to(DEVICE)
                        labels = labels.to(DEVICE)
                        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                            enabled=settings["training_settings"]["use_amp"]):
                            logits = model(data_train)

                    # Reshape and compute loss
                    if n_members > 1:
                        # assume logits shape [members, batch, classes]
                        logits = logits.contiguous().view(logits.shape[0] * logits.shape[1], -1)
                        labels = labels.repeat(n_members)
                    
                    
                    loss = criterion(logits, labels)

                    # Scale by n_members so total gradient magnitude matches
                    if settings["data_settings"]["data_set"] == "SST2":
                        loss = loss * n_members

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"],
                                                   settings["training_settings"]["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()

                    gradient_updates += 1
                    if settings["training_settings"]["lr_schedule_name"] != "epoch_step":
                        lr_schedule.step()

                    train_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})

                    if gradient_updates >= settings["training_settings"]["max_steps"]:
                        finish_training = True
                        break

                if settings["training_settings"]["lr_schedule_name"] == "epoch_step":
                    lr_schedule.step()

        # Validation
        if settings["evaluation_settings"]["evaluation"]:
            model.eval()
            all_probs = []
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    max_iter = settings.get("data_settings", {}).get("subset_evaluation_iterations")
                    if max_iter and batch_idx >= max_iter:
                        break

                    if settings["data_settings"]["data_set"] == "SST2":
                        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                        outputs = model(**{k: inputs[k] for k in inputs if k != 'labels'})
                        # logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        logits = outputs
                        labels = inputs['labels'].cpu().numpy()
                    else:
                        data_val, labels = batch
                        data_val = data_val.to(DEVICE)
                        logits = model(data_val)
                        labels = labels.numpy()

                    # Ensemble averaging
                    if n_members > 1:
                        logits = logits.mean(dim=0)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1).cpu().numpy()

                    all_probs.append(probs.cpu())
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            # Metrics
            all_probs = torch.cat(all_probs, dim=0)
            labels_tensor = torch.tensor(all_labels).to(all_probs.device)
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            ece_val, _, _, _, _ = _ECELoss().forward(all_probs.log(), labels_tensor, plot=False)

            print(f"Epoch {epoch+1}: Val Acc: {accuracy:.4f}, ECE: {ece_val.item():.4f}")

        if finish_training:
            break

    print("Training complete.")


def evaluate_snapshot(settings: dict) -> Tuple[float, float, float, float, float]:
    # Snapshot evaluation (unchanged)
    model_name = settings['data_settings']['result_file_name']
    tmp = const.STORAGE_DIR.joinpath('tmp', model_name)
    labels = torch.from_numpy(np.load(tmp/f"{model_name}_labels.npy")).long()
    logits_list = []
    for m in range(settings['model_settings']['nr_members']):
        arr = np.load(tmp/f"{model_name}_logits_snapshot{m}.npy")
        logits_list.append(arr)
    stacked = np.stack(logits_list)
    avg_logits = stacked.mean(0)

    probs = nn.Softmax(dim=2)(torch.from_numpy(stacked)).mean(0)
    ece, _, _, accuracy, _ = _ECELoss().forward(torch.tensor(avg_logits).to(DEVICE), labels.to(DEVICE), plot=True)
    preds = np.argmax(probs.numpy(), 1)
    f1 = f1_score(labels, preds, average='macro')
    prec = precision_score(labels, preds, average='macro')
    rec = recall_score(labels, preds, average='macro')

    return accuracy, f1, prec, rec, ece
