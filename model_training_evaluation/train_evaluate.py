#!/usr/bin/env python

"""
Implements the training pipeline for this project
"""

### IMPORTS ###
# Built-in imports
import time
import shutil
from typing import Tuple

# Lib imports
import torch
import torch.nn as nn
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

### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### STATIC FUNCTIONS ###
def train_evaluate_ensemble(settings: dict, batch_mode: BatchMode = BatchMode.DEFAULT) -> None:
    """
    Train ViT model.

    Parameters
    ----------
    settings : dict
        Experiment settings as well as any other information passed on from loading
    batch_mode : BatchMode, optional
        The batch mode to use, by default BatchMode.DEFAULT
        This encodes whether the data is repeated in the batch dimension
        to train all ensemble members on the same data or if the data is split between the ensemble members.
    """

    # Get model
    model = settings["model_settings"]["model"]

    # Get number of ensemble members
    n_members = settings["model_settings"]["nr_members"]

    # move loss function weights to device (GPU if possible)
    settings["training_settings"]["loss"].weight = settings["training_settings"]["loss"].weight.to(DEVICE)

    # optimizer settings
    optimizer = settings["training_settings"]["optimizer"]

    # learning 
    lr_schedule = settings["training_settings"]["lr_schedule"]

    # initialize loss function
    criterion = settings["training_settings"]["loss"]

    # get data loaders
    if settings["training_settings"]["training"]:
        train_dataloader = settings['training_settings']["training_dataloader"]
    if settings["evaluation_settings"]["evaluation"]:
        val_data_loader = settings['evaluation_settings']["evaluation_dataloader"]

    # get first 1 batch of training data, assume sst2
    print("sanity check")
    for batch_idx, target_params in enumerate(train_dataloader):
        if batch_idx == 0:
            batch = {key: val.to(DEVICE) for key, val in target_params.items()}
            target = batch['labels']
            print("train target shape", target.shape)
            print(target)
    for batch_idx, target_params in enumerate(val_data_loader):
        if batch_idx == 0:
            batch = {key: val.to(DEVICE) for key, val in target_params.items()}
            target = batch['labels']
            print("train target shape", target.shape)
            print(target)


    # number of training steps
    gradient_updates = 0
    finish_training = False

    # initial validation accuracy
    best_val_accuracy = 0

    # Mixed precision grad scaler
    scaler = torch.cuda.amp.GradScaler(enabled=settings["training_settings"]["use_amp"])

    # Create lists for storing times
    train_time_list = []
    inferece_time_list = []

    # training loop
    for epoch in range(settings["training_settings"]["max_epochs"]):
        # initialize train loss of current epoch
        train_loss = 0

        # set model to training mode
        model.train()

        # Get the start time of the epoch
        epoch_start_time = time.time()


        # If training is enabled
        if settings["training_settings"]["training"]:
            # iterate over training batches
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Epoch", position=0) as pbar:
                for batch_idx, target_params in pbar:

                    #if batch_idx==1:
                    #    break

                    if settings["data_settings"]["data_set"] != "SST2":
                        data_train, target = target_params
                    else:
                        batch = target_params
                    train_params = {}

                    # set optimizers gradients to zero
                    optimizer.zero_grad()

                    # move training data and labels to device (GPU if possible)
                    if settings["data_settings"]["data_set"] != "SST2":
                        data_train = data_train.to(DEVICE)
                    else:
                        batch = {key: val.to(DEVICE) for key, val in target_params.items()}
                        target = batch['labels']

                    # forward pass
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                        enabled=settings["training_settings"]["use_amp"]):  # Automatic mixed precision

                        # Assert that the input does not contain NaN or infinite values
                        # assert not torch.isnan(data_train).any(), "Input contains NaN values"
                        # assert not torch.isinf(data_train).any(), "Input contains infinite values"

                        # Forward pass through the model
                        if settings["data_settings"]["data_set"] != "SST2":
                            output = model(data_train)
                        else:
                            output = model(batch)

                        # Assert that the output does not contain NaN or infinite values
                        assert not torch.isnan(output).any(), "Output contains NaN values"
                        assert not torch.isinf(output).any(), "Output contains infinite values"

                        # Reshape the output back into batch dimension for backpropagation
                        output = output.contiguous().view(output.shape[1] * n_members, -1)

                        
                        # Repeat the target for each member to ensure independent training
                        target = target.repeat(n_members)

                        # move target tensor to device (GPU if possible)
                        target = target.to(DEVICE)
                    

                        # calculate training loss
                        loss = criterion(output, target)

                    # backward pass with amp
                    scaler.scale(loss).backward()

                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(settings["training_settings"]["optimizer"].param_groups[0]["params"],
                                                   settings["training_settings"]["gradient_clip"])

                    # update weights
                    scaler.step(optimizer)
                    scaler.update()

                    # count number of gradient updates
                    gradient_updates += 1

                    # update learning rate
                    if settings["training_settings"]["lr_schedule_name"] != "epoch_step":
                        lr_schedule.step()

                    # sum up loss of epoch
                    train_loss += loss.item()

                    # Update the progress bar stats
                    train_params["loss"] = loss.item()
                    pbar.set_postfix(train_params)
                    pbar.update()

                    # maximum number of steps reached
                    if gradient_updates == (settings["training_settings"]["max_steps"]):
                        finish_training = True
                        break

                    # maximum number of steps per epoch reached
                    if "subset_train_iterations" in settings["data_settings"].keys():
                        if batch_idx == (settings["data_settings"]["subset_train_iterations"] - 1):
                            break

            # Get the end time of the epoch
            epoch_end_time = time.time()

            # Calculate and store the epochs training time
            epoch_train_time = epoch_end_time - epoch_start_time
            train_time_list.append(epoch_train_time)

            # update learning rate
            if settings["training_settings"]["lr_schedule_name"] == "epoch_step":
                lr_schedule.step()

       # --- VALIDATION (minimal changes) ---
        if settings["evaluation_settings"]["evaluation"]:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                predictions = np.array([])
                labels = np.array([])
                logits_prediction = torch.tensor([]).to(DEVICE)
                softmax = nn.Softmax(dim=2)

                for batch_idx, target_params in enumerate(val_data_loader):
                    # prepare validation batch
                    if settings["data_settings"]["data_set"] != "SST2":
                        data_val, target = target_params
                        data_val = data_val.to(DEVICE)
                    else:
                        batch = {k: v.to(DEVICE) for k, v in target_params.items()}
                        target = batch['labels']

                    # forward using same pattern as training
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                        enabled=settings["training_settings"]["use_amp"]):
                        if settings["data_settings"]["data_set"] != "SST2":
                            raw = model(data_val)
                        else:
                            raw = model(batch)
                        # reshape to [members, batch, classes]
                        output = raw.view(n_members, -1, raw.shape[-1])

                        # member-wise softmax and average
                        member_probs = softmax(output)            # [members, batch, classes]
                        ensemble_probs = member_probs.mean(dim=0)  # [batch, classes]
                        # assert torch.allclose(ensemble_probs.sum(dim=1),
                        #                       torch.ones(ensemble_probs.size(0), device=DEVICE),
                        #                       atol=1e-5), "Output is not a probability distribution"

                        
                        #print(member_probs.shape)
                        #print(ensemble_probs.shape)
                        # log-probs and loss
                        log_ens = torch.log(ensemble_probs)
                        val_loss += nn.NLLLoss(weight=criterion.weight)(log_ens, target.to(DEVICE))

                    #print(ensemble_probs)
                    # collect preds and labels
                    preds = torch.argmax(ensemble_probs, dim=1)
                    logits_prediction = torch.cat((logits_prediction, output.mean(dim=0)))
                    labels = np.concatenate((labels, target.cpu().numpy().flatten()))
                    predictions = np.concatenate((predictions, preds.cpu().numpy().flatten()))

                    if batch_idx + 1 == settings.get("data_settings", {}).get("subset_evaluation_iterations", float('inf')):
                        break


                # calculate ECE
                if DEVICE == 'cuda':
                    ece_criterion = _ECELoss().cuda()
                else:
                    ece_criterion = _ECELoss()

                # If training is finished plot a reliability diagram
                plot_ece = False
                file_name = None
                if finish_training or epoch == (settings["training_settings"]["max_epochs"] - 1):
                    plot_ece = True
                    file_name = "reliability_diagram_{}".format(settings['data_settings']['result_file_name'])

                # Calculate ECE
                ece, accs, confs, accuracy, avg_conf = ece_criterion.forward(logits_prediction,
                                                                             torch.tensor(labels).to(DEVICE),
                                                                             plot=plot_ece, file_name=file_name)

        # Calculate metrics
        f1 = f1_score(labels, predictions, average='macro')
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        if n_members > 1:
            # dummy value
            disagreement = 0
            distance_predicted_distributions = 0
            #disagreement = (2 / (n_members * (n_members - 1))) * disagreement * (1 / len(val_data_loader.dataset))
            #distance_predicted_distributions = (2 / (n_members * (n_members - 1))) * distance_predicted_distributions

        if ("cross_validation" in settings["training_settings"].keys() and
                settings["training_settings"]["cross_validation"]):
            if epoch == 0:
                # Save stats
                model_name = settings["data_settings"]["result_file_name"]
                stats_name_cross_validation = f"{model_name}_stats_cross_validation.csv"
                const.STATS_DIR.mkdir(parents=True, exist_ok=True)
                stats_path_cross_validation = const.STATS_DIR.joinpath(stats_name_cross_validation)

                # Create stats string
                header_string = ("epoch,train_loss,val_loss,accuracy,f1,precision,recall,ece,disagreement,"
                                 "distance_predicted_distributions\n")
                if settings["training_settings"]["training"]:
                    stats_string = (
                        f"{epoch + 1},{train_loss / len(train_dataloader)},{val_loss / len(val_data_loader)},{accuracy},{f1},"
                        f"{precision},{recall},{ece[0]},{disagreement},{distance_predicted_distributions}\n")
                else:
                    stats_string = (f"{epoch + 1},{None},{val_loss / len(val_data_loader)},{accuracy},{f1},"
                                    f"{precision},{recall},{ece[0]},{disagreement},{distance_predicted_distributions}\n")

                # Write stats to file
                with open(stats_path_cross_validation, "w") as stats_file:
                    stats_file.write(header_string)
                    stats_file.write(stats_string)
            else:
                # append stats to existing csv file
                stats_string = (
                    f"{epoch + 1},{train_loss / len(train_dataloader)},{val_loss / len(val_data_loader)},{accuracy},{f1},"
                    f"{precision},{recall},{ece[0]},{disagreement},{distance_predicted_distributions}\n")
                with open(stats_path_cross_validation, "a") as stats_file:
                    stats_file.write(stats_string)

            # Save predictions for later aggregation
            model_name = settings["data_settings"]["model_name"]
            tmp_dir = const.STORAGE_DIR.joinpath("tmp", model_name)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            predictions_file = tmp_dir.joinpath(f"{model_name}_predictions_epoch{epoch + 1}.npy")
            logits_prediction_file = tmp_dir.joinpath(f"{model_name}_logits_epoch{epoch + 1}.npy")

            # Save labels only once (they are the same for all epochs)
            if epoch == 0:
                labels_file = tmp_dir.joinpath(f"{model_name}_labels.npy")

            # Save predictions
            if settings["training_settings"]["cross_validation_fold"] == 1:
                # On the first fold, save the predictions to a new file
                np.save(predictions_file, predictions)
                np.save(logits_prediction_file, logits_prediction.cpu().detach().numpy())
                if epoch == 0:
                    np.save(labels_file, labels)
            else:
                # On the other folds, append the predictions to the existing file
                # Predictions are saved as numpy binary files
                with NpyAppendArray(predictions_file) as predictions_array:
                    predictions_array.append(predictions)
                with NpyAppendArray(logits_prediction_file) as logits_array:
                    logits_array.append(logits_prediction.cpu().detach().numpy())
                if epoch == 0:
                    with NpyAppendArray(labels_file) as labels_array:
                        labels_array.append(labels)

        # Only write to tensorboard and save model if training is enabled
        if settings["training_settings"]["training"]:
            # print loss scores
            print(
                f'Training: Epoch [{epoch + 1}/{settings["training_settings"]["max_epochs"]}], Loss: {train_loss / len(train_dataloader)}, LR: {optimizer.param_groups[0]["lr"]}')
            print(
                f'Validation: Epoch [{epoch + 1}/{settings["training_settings"]["max_epochs"]}], Loss: {val_loss / len(val_data_loader)}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, ECE: {ece[0]}, Disagreement: {disagreement}, Distance predicted distributions: {distance_predicted_distributions}')

            # log the running loss
            if settings["data_settings"]["tensorboard"] is True:
                settings["data_settings"]["tensorboard_writer"].add_scalar('Training loss',
                                                                           train_loss / len(train_dataloader),
                                                                           epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation loss',
                                                                           val_loss / len(val_data_loader), epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Accuracy', accuracy, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Average conf', avg_conf,
                                                                           epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation F1', f1, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Precision', precision, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation Recall', recall, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Validation ECE', ece[0], epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Disagreement', disagreement, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].add_scalar('Distance predicted distributions',
                                                                           distance_predicted_distributions, epoch + 1)
                settings["data_settings"]["tensorboard_writer"].close()

            # Save the model (based on early stopping on evaluation accuracy)
            if settings["training_settings"]["early_stopping"] is True:
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    model_name = settings["data_settings"]["result_file_name"]
                    save_name = f"{model_name}.pt"
                    save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)
                    model_state_dict = settings["model_settings"]["model_params"]
                    const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(model_state_dict, save_path)
                    print(f"Model saved to {save_path} with accuracy {accuracy} after {gradient_updates} iterations")

            if "mode" in settings["training_settings"].keys() and settings["training_settings"]["mode"] == "snapshot":
                if lr_schedule.check_cycle_state(gradient_updates):
                    cycle_count = lr_schedule.get_cycle_count(gradient_updates)

                    model_name = settings["data_settings"]["result_file_name"]
                    save_name = f"{model_name}_snapshot{cycle_count}.pt"
                    save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)
                    model_state_dict = settings["model_settings"]["model_params"]
                    const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(model_state_dict, save_path)

                    tmp_dir = const.STORAGE_DIR.joinpath("tmp", model_name)
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    predictions_file = tmp_dir.joinpath(f"{model_name}_predictions_snapshot{cycle_count}.npy")
                    logits_prediction_file = tmp_dir.joinpath(f"{model_name}_logits_snapshot{cycle_count}.npy")
                    labels_file = tmp_dir.joinpath(f"{model_name}_labels.npy")

                    # Save predictions
                    np.save(predictions_file, predictions)
                    np.save(logits_prediction_file, logits_prediction.cpu().detach().numpy())

                    if cycle_count == 0:
                        np.save(labels_file, labels)

                    print(f"\n\nSnapshot model saved to {save_path} with accuracy {accuracy} after {gradient_updates} iterations\n\n")

            # check if maximum number of steps is reached
            if (finish_training and
                    ("mode" not in settings["training_settings"].keys() or settings["training_settings"]["mode"] != "snapshot")):
                # Save final model
                if settings["training_settings"]["early_stopping"] is False:
                    model_name = settings["data_settings"]["result_file_name"]
                    save_name = f"{model_name}.pt"
                    save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)
                    model_state_dict = settings["model_settings"]["model_params"]
                    const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(model_state_dict, save_path)
                    print(f"Model saved to {save_path} with accuracy {accuracy} after {gradient_updates} iterations")

        # Print validation metrics if training is disabled
        else:
            print(
                f'Validation: Epoch [{epoch + 1}/{settings["training_settings"]["max_epochs"]}], Loss: {val_loss / len(val_data_loader)}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, ECE: {ece[0]}, Disagreement: {disagreement}, Distance predicted distributions: {distance_predicted_distributions}')

        if not settings["training_settings"]["training"]:
            # If model is not training only run through the loop once
            break

    if "mode" in settings["training_settings"].keys() and settings["training_settings"]["mode"] == "snapshot":
        accuracy, f1, precision, recall, ece, nll, brier_score_sum = evaluate_snapshot(settings)
        train_loss, val_loss = np.nan, np.nan
        disagreement, distance_predicted_distributions = np.nan, np.nan

    # Save stats
    model_name = settings["data_settings"]["result_file_name"]
    stats_name = f"{model_name}_stats.csv"
    const.STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = const.STATS_DIR.joinpath(stats_name)

    # Create stats string
    header_string = ("train_loss,val_loss,accuracy,f1,precision,recall,ece,disagreement,"
                     "distance_predicted_distributions\n")
    if settings["training_settings"]["training"] is True:
        stats_string = (f"{train_loss / len(train_dataloader)},{val_loss / len(val_data_loader)},{accuracy},{f1},"
                        f"{precision},{recall},{ece[0]},{disagreement},{distance_predicted_distributions}\n")
    else:
        stats_string = (f"{None},{val_loss / len(val_data_loader)},{accuracy},{f1},"
                        f"{precision},{recall},{ece[0]},{disagreement},{distance_predicted_distributions}\n")

    if "NLL_Brier_Score" in settings["evaluation_settings"].keys() and settings["evaluation_settings"][
        "NLL_Brier_Score"] is True:
        header_string_NLL_Brier = ("train_loss,val_loss,accuracy,f1,precision,recall,ece,disagreement,"
                                   "distance_predicted_distributions, NLL, Brier\n")
        stats_string_NLL_Brier = (f"{None},{val_loss / len(val_data_loader)},{accuracy},{f1},"
                                  f"{precision},{recall},{ece[0]},{disagreement},{distance_predicted_distributions},{nll / val_data_loader.dataset.__len__()},{brier_score_sum / val_data_loader.dataset.__len__()}\n")
        # Write stats to file
        model_name = settings["data_settings"]["result_file_name"]
        stats_name_nll_brier = f"{model_name}_statsNLLBrier.csv"
        const.STATS_DIR.mkdir(parents=True, exist_ok=True)
        stats_path_nll_brier = const.STATS_DIR.joinpath(stats_name_nll_brier)
        with open(stats_path_nll_brier, "w") as stats_file:
            stats_file.write(header_string_NLL_Brier)
            stats_file.write(stats_string_NLL_Brier)

    # Write stats to file
    with open(stats_path, "w") as stats_file:
        stats_file.write(header_string)
        stats_file.write(stats_string)

    # Calculate average times
    avg_train_time = np.mean(train_time_list)
    avg_inference_time = np.mean(inferece_time_list)

    # Print average times
    print("Average training time per epoch: ", avg_train_time)
    print("Average inference time per batch: ", avg_inference_time)


def evaluate_snapshot(settings: dict) -> Tuple[float, float, float, float, float]:
    """
    Evaluate a snapshot model.

    Parameters
    ----------
    settings : dict
        Experiment settings as well as any other information passed on from loading
    model_path : str
        Path to the model to evaluate

    Returns
    -------
    """

    model_name = settings["data_settings"]["result_file_name"]

    tmp_dir = const.STORAGE_DIR.joinpath("tmp", model_name)

    # Load the labels
    labels_file = tmp_dir.joinpath(f"{model_name}_labels.npy")
    labels = np.load(labels_file)
    labels = torch.from_numpy(labels)
    labels = labels.type(torch.LongTensor)

    logits_stacked = []

    for cycle in range(settings["model_settings"]["nr_members"]):
        # Load logit predictions
        logits_prediction_file = tmp_dir.joinpath(f"{model_name}_logits_snapshot{cycle}.npy")
        logits_predictions = np.load(logits_prediction_file)
        logits_stacked.append(logits_predictions)

    logits_stacked = np.stack(logits_stacked)

    logits_avg = logits_stacked.mean(axis=0)

    softmax = nn.Softmax(dim=2)
    output_softmax = softmax(torch.from_numpy(logits_stacked)).mean(dim=0)
    output_log_softmax = torch.log(output_softmax)

    predictions_avg = np.argmax(output_softmax, axis=1)

    # Define the ECE criterion
    if DEVICE == 'cuda':
        # ece_criterion = _ECELoss(multi_label=settings["data_settings"]["multi_label"]).cuda()
        ece_criterion = _ECELoss().cuda()
    else:
        # ece_criterion = _ECELoss(multi_label=settings["data_settings"]["multi_label"])
        ece_criterion = _ECELoss()

    # Construct the reliability diagram filename
    reliability_diagram_name = "reliability_diagram_{}".format(
        settings['data_settings']['result_file_name'])
    # Calculate the ECE and plot the reliability diagram
    ece, accs, confs, accuracy, avg_conf = ece_criterion.forward(torch.tensor(logits_avg).to(DEVICE),
                                                                 torch.tensor(labels).to(DEVICE),
                                                                 plot=True,
                                                                 file_name=reliability_diagram_name)

    # Calculate the f1, precision and recall
    f1 = f1_score(labels, predictions_avg, average='macro')
    precision = precision_score(labels, predictions_avg, average='macro')
    recall = recall_score(labels, predictions_avg, average='macro')

    if "NLL_Brier_Score" in settings["evaluation_settings"].keys() and \
            settings["evaluation_settings"]["NLL_Brier_Score"] is True:
        # calculate NLL loss
        NLL = nn.NLLLoss(reduction="sum")
        nll = NLL(output_log_softmax, labels)

        # calculate brier score
        brier_score = torch.sum(
            (output_softmax.cpu() - torch.eye(output_softmax.shape[1])[labels.cpu()]) ** 2)
    else:
        nll = np.nan
        brier = np.nan

    # shutil.rmtree(tmp_dir)

    return accuracy, f1, precision, recall, ece, nll, brier_score