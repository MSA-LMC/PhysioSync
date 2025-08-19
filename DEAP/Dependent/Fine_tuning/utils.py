import torch
import numpy as np
import os
from torch.utils import data
from models import Encoder_eeg, Encoder_gsr

def prepare_data(datasets, datasets_gsr, labels, train_list, val_list, args):
    """
    Prepare the EEG and GSR data for training and testing.

    Args:
        datasets (np.array): EEG data.
        datasets_gsr (np.array): GSR data.
        labels (np.array): Labels for the data.
        train_list (list): List of training indices.
        val_list (list): List of validation indices.
        args (Namespace): Arguments containing batch size ('B').

    Returns:
        train_loader, test_loader: DataLoader objects for training and testing.
    """
    # Select training and testing data based on provided indices
    x_train_eeg = datasets[list(train_list), :, :, :]
    x_train_gsr = datasets_gsr[list(train_list), :, :, :]
    y_train_eeg = labels[list(train_list)]

    x_test_eeg = datasets[list(val_list), :, :, :]
    x_test_gsr = datasets_gsr[list(val_list), :, :, :]
    y_test_eeg = labels[list(val_list)]

    # Reshape the data to match expected input format
    x_train_gsr = x_train_gsr.reshape(-1, 1, x_train_gsr.shape[-2], x_train_gsr.shape[-1])
    x_train_eeg = x_train_eeg.reshape(-1, 1, x_train_eeg.shape[-2], x_train_eeg.shape[-1])
    x_test_gsr = x_test_gsr.reshape(-1, 1, x_test_gsr.shape[-2], x_test_gsr.shape[-1])
    x_test_eeg = x_test_eeg.reshape(-1, 1, x_test_eeg.shape[-2], x_test_eeg.shape[-1])

    # Flatten labels
    y_train = y_train_eeg.reshape(-1, )
    y_test = y_test_eeg.reshape(-1, )

    # Create TensorDataset and DataLoader for train and test sets
    train_dataset = data.TensorDataset(torch.tensor(x_train_eeg), torch.tensor(x_train_gsr), torch.tensor(y_train))
    train_loader = data.DataLoader(train_dataset, batch_size=args.B, shuffle=True)

    test_dataset = data.TensorDataset(torch.tensor(x_test_eeg), torch.tensor(x_test_gsr), torch.tensor(y_test))
    test_loader = data.DataLoader(test_dataset, batch_size=args.B, shuffle=True)

    return train_loader, test_loader

def load_models(fold, best_epoch, best_epoch_1s, root_dir_5s, root_dir_1s, args):
    # Get the best pretraining epoch for EEG and GSR
    best_pretrain_epoch_eeg_5s = int(best_epoch['best_epoch_eeg'][fold])
    best_pretrain_epoch_gsr_5s = int(best_epoch['best_epoch_gsr'][fold])
    checkpoint_name_eeg_5s = f'eeg_best_checkpoint_{best_pretrain_epoch_eeg_5s:04d}.pth'
    checkpoint_name_gsr_5s = f'gsr_best_checkpoint_{best_pretrain_epoch_gsr_5s:04d}.pth'

    best_pretrain_epoch_eeg_1s = int(best_epoch_1s['best_epoch_eeg'][fold])
    best_pretrain_epoch_gsr_1s = int(best_epoch_1s['best_epoch_gsr'][fold])
    checkpoint_name_eeg_1s = f'eeg_best_checkpoint_{best_pretrain_epoch_eeg_1s:04d}.pth'
    checkpoint_name_gsr_1s = f'gsr_best_checkpoint_{best_pretrain_epoch_gsr_1s:04d}.pth'

    # Load the checkpoints from the specified directories
    checkpoint_eeg_5s = torch.load(os.path.join(root_dir_5s, str(fold), checkpoint_name_eeg_5s), map_location=args.device)
    checkpoint_gsr_5s = torch.load(os.path.join(root_dir_5s, str(fold), checkpoint_name_gsr_5s), map_location=args.device)

    checkpoint_eeg_1s = torch.load(os.path.join(root_dir_1s, str(fold), checkpoint_name_eeg_1s), map_location=args.device)
    checkpoint_gsr_1s = torch.load(os.path.join(root_dir_1s, str(fold), checkpoint_name_gsr_1s), map_location=args.device)

    # Initialize the models and load the state dicts into the models
    model_eeg_5s = Encoder_eeg(embed_dim=args.embed_dim_5s, eeg_dim=args.eeg_dim_5s, mlp_dim=args.mlp_dim_5s)
    model_eeg_5s.load_state_dict(checkpoint_eeg_5s['state_dict'], strict=False)
    model_eeg_5s.to(args.device)

    model_eeg_1s = Encoder_eeg(embed_dim=args.embed_dim_1s, eeg_dim=args.eeg_dim_1s, mlp_dim=args.mlp_dim_1s)
    model_eeg_1s.load_state_dict(checkpoint_eeg_1s['state_dict'], strict=False)
    model_eeg_1s.to(args.device)

    model_gsr_5s = Encoder_gsr(embed_dim=args.embed_dim_5s, gsr_dim=args.gsr_dim_5s, mlp_dim=args.mlp_dim_5s)
    model_gsr_5s.load_state_dict(checkpoint_gsr_5s['state_dict'], strict=False)
    model_gsr_5s.to(args.device)

    model_gsr_1s = Encoder_gsr(embed_dim=args.embed_dim_1s, gsr_dim=args.gsr_dim_1s, mlp_dim=args.mlp_dim_1s)
    model_gsr_1s.load_state_dict(checkpoint_gsr_1s['state_dict'], strict=False)
    model_gsr_1s.to(args.device)

    return model_eeg_5s, model_eeg_1s, model_gsr_5s, model_gsr_1s

def matrix_percent(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    label_to_index = {label: i for i, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:

            confusion_matrix[label_to_index[true], label_to_index[pred]] += 1
    class_totals = confusion_matrix.sum(axis=1, keepdims=True)
    class_totals[class_totals == 0] = 1
    confusion_matrix_percent = confusion_matrix / class_totals * 100
    return confusion_matrix_percent