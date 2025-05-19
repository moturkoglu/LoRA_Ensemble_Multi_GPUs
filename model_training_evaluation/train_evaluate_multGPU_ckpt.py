#!/usr/bin/env python

"""
Implements the training pipeline for this project with minimal DDP changes
"""

### IMPORTS ###
import time
import shutil
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray

# Custom imports
from utils_uncertainty import _ECELoss, function_space_analysis
from models.lora_ensemble import BatchMode
from models.model_loader import load_model
from utils_GPU import DEVICE
import const

### HELPER FUNCTION ###
def get_safe_state_dict(model: torch.nn.Module) -> dict:
    """
    Returns the state dict of the underlying model safely,
    handling DDP wrappers.
    """
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

### DDP INITIALIZATION (only once at the start) ###
operating_system = "linux"
operating_system = "windows"
if operating_system == "linux":
    dist.init_process_group(backend="nccl")
elif operating_system == "windows":
    dist.init_process_group(backend="gloo")  
local_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


def train_evaluate_ensemble(settings: dict, batch_mode: BatchMode = BatchMode.DEFAULT) -> None:
    """
    Train ViT model (DDP version with minimal changes).
    """

    # ----------------------------------------------------------------
    # 1) GET MODEL & WRAP IN DDP
    # ----------------------------------------------------------------
    model = settings["model_settings"]["model"].to(local_rank)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params} | Trainable params: {trainable_params}")

    # Wrap the model for DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    n_members = settings["model_settings"]["nr_members"]

    # Move loss function weights to the correct GPU
    settings["training_settings"]["loss"].weight = settings["training_settings"]["loss"].weight.to(local_rank)
    criterion = settings["training_settings"]["loss"]

    # ----------------------------------------------------------------
    # 2) OPTIMIZER, LR SCHEDULER, ETC.
    # ----------------------------------------------------------------
    optimizer = settings["training_settings"]["optimizer"]
    lr_schedule = settings["training_settings"]["lr_schedule"]

    # Mixed precision grad scaler
    scaler = torch.cuda.amp.GradScaler(enabled=settings["training_settings"]["use_amp"])

    # ----- MODIFIED: Check for resume checkpoint -----
    start_epoch = 0
    if settings["training_settings"].get("checkpoint_file", None):
        checkpoint = torch.load(settings["training_settings"]["checkpoint_file"],
                                map_location=torch.device("cuda", local_rank))
        ckpt_state = checkpoint["model_state_dict"]
        # Adjust keys: if they do not start with "module.", add the prefix
        if not list(ckpt_state.keys())[0].startswith("module."):
            ckpt_state = {"module." + k: v for k, v in ckpt_state.items()}
        model.load_state_dict(ckpt_state)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_schedule.load_state_dict(checkpoint["lr_schedule_state_dict"])
        # Restore AMP scaler state if available
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # Restore RNG states if saved (commented out)
        # if "rng_state" in checkpoint:
        #     torch.set_rng_state(checkpoint["rng_state"])
        #     if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        #         torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        gradient_updates = checkpoint["gradient_updates"]
        print(f"Resuming training from epoch {start_epoch} with {gradient_updates} gradient updates")
    else:
        gradient_updates = 0
    # ----- END MODIFIED -----

    # ----------------------------------------------------------------
    # 3) CREATE DISTRIBUTED SAMPLERS & DATALOADERS
    # ----------------------------------------------------------------
    train_dataset = settings["training_settings"]["training_dataset"]
    val_dataset = settings["evaluation_settings"]["evaluation_dataset"]

    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=settings["training_settings"]["training_batch_size"],
        sampler=train_sampler,
        num_workers=settings["data_settings"]["num_workers"],
        pin_memory=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=False
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=settings["evaluation_settings"]["evaluation_batch_size"],
        sampler=val_sampler,
        num_workers=settings["data_settings"]["num_workers"],
        pin_memory=True
    )

    finish_training = False
    best_val_accuracy = 0

    # Lists for storing timing information
    train_time_list = []
    inferece_time_list = []

    # ----------------------------------------------------------------
    # 4) TRAINING LOOP
    # ----------------------------------------------------------------
    for epoch in range(start_epoch, settings["training_settings"]["max_epochs"]):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss = 0.0
        model.train()
        epoch_start_time = time.time()

        if settings["training_settings"]["training"]:
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Epoch", position=0,
                      disable=(local_rank != 0)) as pbar:
                for batch_idx, (data_train, target) in pbar:
                    train_params = {}
                    optimizer.zero_grad()
                    data_train = data_train.to(local_rank, non_blocking=True)
                    target = target.to(local_rank, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.float16,
                                        enabled=settings["training_settings"]["use_amp"]):
                        output = model(data_train)
                        # Reshape output: from [n_members, B, classes] to [B * n_members, classes]
                        output = output.contiguous().view(output.shape[1] * n_members, -1)
                        target = target.repeat(n_members)
                        loss = criterion(output, target)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        optimizer.param_groups[0]["params"],
                        settings["training_settings"]["gradient_clip"]
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    gradient_updates += 1

                    if settings["training_settings"]["lr_schedule_name"] != "epoch_step":
                        lr_schedule.step()

                    train_loss += loss.item()

                    train_params["loss"] = loss.item()
                    pbar.set_postfix(train_params)
                    pbar.update()

                    if gradient_updates == settings["training_settings"]["max_steps"]:
                        finish_training = True
                        break

                    if "subset_train_iterations" in settings["data_settings"]:
                        if batch_idx == (settings["data_settings"]["subset_train_iterations"] - 1):
                            break

            epoch_end_time = time.time()
            train_time_list.append(epoch_end_time - epoch_start_time)

            if settings["training_settings"]["lr_schedule_name"] == "epoch_step":
                lr_schedule.step()

        # ----------------------------------------------------------------
        # 5) VALIDATION
        # ----------------------------------------------------------------
        if settings["evaluation_settings"]["evaluation"]:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                predictions = np.array([])
                labels = np.array([])
                logits_prediction = torch.tensor([]).to(local_rank)
                disagreement = 0
                distance_predicted_distributions = 0
                nll = 0
                brier_score_sum = 0

                for batch_idx, (data_val, target) in enumerate(val_data_loader):
                    print(batch_idx, "of", len(val_data_loader))
                    data_val = data_val.to(local_rank, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16,
                                        enabled=settings["training_settings"]["use_amp"]):
                        if settings["model_settings"]["ensemble_type"] == "LoRA_Former" \
                           or not settings["evaluation_settings"]["timing"]:
                            inference_start_time = time.time()
                            output = model(data_val)
                            inference_end_time = time.time()
                            inferece_time_list.append(inference_end_time - inference_start_time)
                        else:
                            inference_member_list = []
                            output_list = []
                            for member in model.vit_models:
                                inference_start_time = time.time()
                                output = member(data_val)
                                inference_end_time = time.time()
                                inference_member_list.append(inference_end_time - inference_start_time)
                                output_list.append(output)
                            inferece_time_list.append(inference_member_list)
                            output = torch.stack(output_list)

                        softmax = nn.Softmax(dim=2)
                        output_softmax = softmax(output).mean(dim=0)
                        output_log_softmax = torch.log(output_softmax)
                        output_all_mean = output.mean(dim=0)
                        output_disagreement = output

                        target = target.to(local_rank, non_blocking=True)
                        val_criterion = nn.NLLLoss(weight=criterion.weight)
                        val_loss += val_criterion(output_log_softmax, target)

                        if ("NLL_Brier_Score" in settings["evaluation_settings"]
                                and settings["evaluation_settings"]["NLL_Brier_Score"]):
                            NLL = nn.NLLLoss(reduction="sum")
                            nll += NLL(output_log_softmax, target)
                            brier_score = torch.sum(
                                (output_softmax.cpu() - torch.eye(output_softmax.shape[1])[target.cpu()]) ** 2
                            )
                            brier_score_sum += brier_score

                    prediction = torch.argmax(output_softmax, dim=1)
                    logits_prediction = torch.cat((logits_prediction.to(local_rank), output_all_mean.to(local_rank)))
                    labels = np.concatenate((labels, target.cpu().numpy().flatten()))
                    predictions = np.concatenate((predictions, prediction.cpu().numpy().flatten()))

                    if n_members > 1:
                        if DEVICE == 'cuda':
                            function_space = function_space_analysis().cuda()
                        else:
                            function_space = function_space_analysis()
                        for i in range(n_members):
                            for j in range(i + 1, n_members):
                                disagreement_pred, distance_pred = function_space.forward(
                                    output_disagreement[i], output_disagreement[j]
                                )
                                disagreement += disagreement_pred
                                distance_predicted_distributions += distance_pred

                    if "subset_evaluation_iterations" in settings["data_settings"]:
                        if batch_idx == (settings["data_settings"]["subset_evaluation_iterations"] - 1):
                            break

                if DEVICE == 'cuda':
                    ece_criterion = _ECELoss().cuda()
                else:
                    ece_criterion = _ECELoss()

                plot_ece = False
                file_name = None
                if finish_training or epoch == (settings["training_settings"]["max_epochs"] - 1):
                    plot_ece = True
                    file_name = f"reliability_diagram_{settings['data_settings']['result_file_name']}"

                ece, accs, confs, accuracy, avg_conf = ece_criterion.forward(
                    logits_prediction, torch.tensor(labels).to(local_rank),
                    plot=plot_ece, file_name=file_name
                )
        else:
            val_loss = 0
            accuracy = 0
            avg_conf = 0
            disagreement = 0
            distance_predicted_distributions = 0

        # ----------------------------------------------------------------
        # 6) METRICS & LOGGING
        # ----------------------------------------------------------------
        if settings["evaluation_settings"]["evaluation"]:
            f1 = f1_score(labels, predictions, average='macro')
            precision = precision_score(labels, predictions, average='macro')
            recall = recall_score(labels, predictions, average='macro')

            if n_members > 1:
                ds = len(val_data_loader.dataset) if len(val_data_loader.dataset) > 0 else 1
                disagreement = (2 / (n_members * (n_members - 1))) * disagreement * (1 / ds)
                distance_predicted_distributions = (2 / (n_members * (n_members - 1))) * distance_predicted_distributions
        else:
            f1, precision, recall = 0, 0, 0

        if settings["training_settings"]["training"]:
            if local_rank == 0:
                avg_train_loss = train_loss / len(train_dataloader) if len(train_dataloader) > 0 else train_loss
                avg_val_loss = val_loss / len(val_data_loader) if len(val_data_loader) > 0 else val_loss

                print(
                    f"Rank {local_rank} - Training: Epoch [{epoch + 1}/{settings['training_settings']['max_epochs']}], "
                    f"Loss: {avg_train_loss}, LR: {optimizer.param_groups[0]['lr']}"
                )
                print(
                    f"Rank {local_rank} - Validation: Epoch [{epoch + 1}/{settings['training_settings']['max_epochs']}], "
                    f"Loss: {avg_val_loss}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, "
                    f"Recall: {recall}, ECE: {ece[0] if isinstance(ece, tuple) else ece}, "
                    f"Disagreement: {disagreement}, Distance predicted distributions: {distance_predicted_distributions}"
                )

            if settings["data_settings"]["tensorboard"] is True and local_rank == 0:
                avg_train_loss = train_loss / len(train_dataloader) if len(train_dataloader) > 0 else train_loss
                avg_val_loss = val_loss / len(val_data_loader) if len(val_data_loader) > 0 else val_loss

                settings["data_settings"]["tensorboard_writer"].add_scalar('Training loss', avg_train_loss, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation loss', avg_val_loss, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Accuracy', accuracy, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Average conf', avg_conf, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation F1', f1, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Precision', precision, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Recall', recall, epoch + 1)
                if isinstance(ece, tuple):
                    ece_val = ece[0]
                else:
                    ece_val = ece
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation ECE', ece_val, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Disagreement', disagreement, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Distance predicted distributions',
                                                                           distance_predicted_distributions, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].close()

            if settings["training_settings"]["early_stopping"] is True and local_rank == 0:
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    model_name = settings["data_settings"]["result_file_name"]
                    save_name = f"{model_name}.pt"
                    save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)
                    model_state_dict = get_safe_state_dict(model)
                    const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(model_state_dict, save_path)
                    print(f"Model saved to {save_path} with accuracy {accuracy} after {gradient_updates} iterations")

            if ("mode" in settings["training_settings"]
                    and settings["training_settings"]["mode"] == "snapshot"
                    and local_rank == 0):
                if lr_schedule.check_cycle_state(gradient_updates):
                    cycle_count = lr_schedule.get_cycle_count(gradient_updates)
                    model_name = settings["data_settings"]["result_file_name"]
                    save_name = f"{model_name}_snapshot{cycle_count}.pt"
                    save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)
                    model_state_dict = get_safe_state_dict(model)
                    const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(model_state_dict, save_path)

                    tmp_dir = const.STORAGE_DIR.joinpath("tmp", model_name)
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    predictions_file = tmp_dir.joinpath(f"{model_name}_predictions_snapshot{cycle_count}.npy")
                    logits_prediction_file = tmp_dir.joinpath(f"{model_name}_logits_snapshot{cycle_count}.npy")
                    labels_file = tmp_dir.joinpath(f"{model_name}_labels.npy")

                    np.save(predictions_file, predictions)
                    np.save(logits_prediction_file, logits_prediction.cpu().detach().numpy())

                    if cycle_count == 0:
                        np.save(labels_file, labels)

                    print(f"\n\nSnapshot model saved to {save_path} with accuracy {accuracy} "
                          f"after {gradient_updates} iterations\n\n")

            if finish_training and ("mode" not in settings["training_settings"]
                                    or settings["training_settings"]["mode"] != "snapshot"):
                if settings["training_settings"]["early_stopping"] is False and local_rank == 0:
                    model_name = settings["data_settings"]["result_file_name"]
                    save_name = f"{model_name}.pt"
                    save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)
                    model_state_dict = get_safe_state_dict(model)
                    const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(model_state_dict, save_path)
                    print(f"Model saved to {save_path} with accuracy {accuracy} after {gradient_updates} iterations")
                    
            # ----- MODIFIED: Checkpoint Saving After Each Epoch -----
            if settings["training_settings"]["training"] and local_rank == 0:
                model_name = settings["data_settings"]["result_file_name"]
                checkpoint_path = const.MODEL_STORAGE_DIR.joinpath(
                    f"{model_name}_checkpoint_epoch_{epoch}.pt")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": get_safe_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_schedule_state_dict": lr_schedule.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),  # Save AMP scaler state
                    "gradient_updates": gradient_updates,
                    # "rng_state": torch.get_rng_state().clone().detach(),  # RNG state saving commented out
                }
                # if torch.cuda.is_available():
                #     checkpoint["cuda_rng_state"] = [state.clone().detach() for state in torch.cuda.get_rng_state_all()]
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")
            # ----- END MODIFIED -----
        else:
            if local_rank == 0:
                avg_val_loss = val_loss / len(val_data_loader) if len(val_data_loader) > 0 else val_loss
                print(f'Validation: Epoch [{epoch + 1}/{settings["training_settings"]["max_epochs"]}], '
                      f'Loss: {avg_val_loss}, Accuracy: {accuracy}, F1: {f1}, '
                      f'Precision: {precision}, Recall: {recall}, ECE: {ece[0] if isinstance(ece, tuple) else ece}, '
                      f'Disagreement: {disagreement}, Distance predicted distributions: {distance_predicted_distributions}')

        if not settings["training_settings"]["training"]:
            break

    if ("mode" in settings["training_settings"] and settings["training_settings"]["mode"] == "snapshot"):
        accuracy, f1, precision, recall, ece, nll, brier_score_sum = evaluate_snapshot(settings)
        train_loss, val_loss = np.nan, np.nan
        disagreement, distance_predicted_distributions = np.nan, np.nan

    if local_rank == 0:
        model_name = settings["data_settings"]["result_file_name"]
        stats_name = f"{model_name}_stats.csv"
        const.STATS_DIR.mkdir(parents=True, exist_ok=True)
        stats_path = const.STATS_DIR.joinpath(stats_name)

        header_string = ("train_loss,val_loss,accuracy,f1,precision,recall,ece,disagreement,"
                         "distance_predicted_distributions\n")

        if settings["training_settings"]["training"] is True and len(train_dataloader) > 0 and len(val_data_loader) > 0:
            stats_string = (f"{train_loss / len(train_dataloader)},{val_loss / len(val_data_loader)},"
                            f"{accuracy},{f1},{precision},{recall},{ece[0] if isinstance(ece, tuple) else ece},"
                            f"{disagreement},{distance_predicted_distributions}\n")
        else:
            stats_string = (f"{None},{None},{accuracy},{f1},{precision},{recall},"
                            f"{ece[0] if isinstance(ece, tuple) else ece},{disagreement},{distance_predicted_distributions}\n")

        if "NLL_Brier_Score" in settings["evaluation_settings"] and settings["evaluation_settings"]["NLL_Brier_Score"]:
            header_string_NLL_Brier = ("train_loss,val_loss,accuracy,f1,precision,recall,ece,disagreement,"
                                       "distance_predicted_distributions,NLL,Brier\n")
            val_len = val_data_loader.dataset.__len__() if len(val_data_loader.dataset) > 0 else 1
            stats_string_NLL_Brier = (
                f"{None},{val_loss / len(val_data_loader) if len(val_data_loader)>0 else val_loss},{accuracy},{f1},"
                f"{precision},{recall},{ece[0] if isinstance(ece, tuple) else ece},{disagreement},"
                f"{distance_predicted_distributions},{nll/val_len},{brier_score_sum/val_len}\n"
            )

            stats_name_nll_brier = f"{model_name}_statsNLLBrier.csv"
            stats_path_nll_brier = const.STATS_DIR.joinpath(stats_name_nll_brier)
            with open(stats_path_nll_brier, "w") as stats_file:
                stats_file.write(header_string_NLL_Brier)
                stats_file.write(stats_string_NLL_Brier)

        with open(stats_path, "w") as stats_file:
            stats_file.write(header_string)
            stats_file.write(stats_string)

        avg_train_time = np.mean(train_time_list) if len(train_time_list) else 0
        avg_inference_time = np.mean(inferece_time_list) if len(inferece_time_list) else 0
        print("Average training time per epoch: ", avg_train_time)
        print("Average inference time per batch: ", avg_inference_time)
