import torch
import numpy as np
import os
from torch.utils import data
from models import Encoder_eeg, Encoder_ecg

def prepare_data(datasets, datasets_ecg, labels, train_list, val_list, args):
    """
    Prepare the EEG and ecg data for training and testing.

    Args:
        datasets (np.array): EEG data.
        datasets_ecg (np.array): ecg data.
        labels (np.array): Labels for the data.
        train_list (list): List of training indices.
        val_list (list): List of validation indices.
        args (Namespace): Arguments containing batch size ('B').

    Returns:
        train_loader, test_loader: DataLoader objects for training and testing.
    """
    # Select training and testing data based on provided indices
    x_train_eeg = datasets[list(train_list), :, :, :]
    x_train_ecg = datasets_ecg[list(train_list), :, :, :]
    y_train_eeg = labels[list(train_list)]

    x_test_eeg = datasets[list(val_list), :, :, :]
    x_test_ecg = datasets_ecg[list(val_list), :, :, :]
    y_test_eeg = labels[list(val_list)]

    # Reshape the data to match expected input format
    x_train_ecg = x_train_ecg.reshape(-1, 1, x_train_ecg.shape[-2], x_train_ecg.shape[-1])
    x_train_eeg = x_train_eeg.reshape(-1, 1, x_train_eeg.shape[-2], x_train_eeg.shape[-1])
    x_test_ecg = x_test_ecg.reshape(-1, 1, x_test_ecg.shape[-2], x_test_ecg.shape[-1])
    x_test_eeg = x_test_eeg.reshape(-1, 1, x_test_eeg.shape[-2], x_test_eeg.shape[-1])

    # Flatten labels
    y_train = y_train_eeg.reshape(-1, )
    y_test = y_test_eeg.reshape(-1, )

    # Create TensorDataset and DataLoader for train and test sets
    train_dataset = data.TensorDataset(torch.tensor(x_train_eeg), torch.tensor(x_train_ecg), torch.tensor(y_train))
    train_loader = data.DataLoader(train_dataset, batch_size=args.B, shuffle=True)

    test_dataset = data.TensorDataset(torch.tensor(x_test_eeg), torch.tensor(x_test_ecg), torch.tensor(y_test))
    test_loader = data.DataLoader(test_dataset, batch_size=args.B, shuffle=True)

    return train_loader, test_loader

def load_models(fold, best_epoch, args):
    # Get the best pretraining epoch for EEG and ecg
    best_pretrain_epoch_eeg = int(best_epoch['best_epoch_eeg'][fold])
    best_pretrain_epoch_ecg = int(best_epoch['best_epoch_ecg'][fold])
    checkpoint_name_eeg = f'eeg_best_checkpoint_{best_pretrain_epoch_eeg:04d}.pth'
    checkpoint_name_ecg = f'ecg_best_checkpoint_{best_pretrain_epoch_ecg:04d}.pth'

    # Load the checkpoints from the specified directories
    checkpoint_eeg = torch.load(os.path.join(args.pth_dir, str(fold), checkpoint_name_eeg), map_location=args.device)
    checkpoint_ecg = torch.load(os.path.join(args.pth_dir, str(fold), checkpoint_name_ecg), map_location=args.device)

    # Initialize the models and load the state dicts into the models
    model_eeg = Encoder_eeg(embed_dim=args.embed_dim, eeg_dim=args.eeg_dim, mlp_dim=args.mlp_dim)
    model_eeg.load_state_dict(checkpoint_eeg['state_dict'], strict=False)
    model_eeg.to(args.device)

    model_ecg = Encoder_ecg(embed_dim=args.embed_dim, ecg_dim=args.ecg_dim, mlp_dim=args.mlp_dim)
    model_ecg.load_state_dict(checkpoint_ecg['state_dict'], strict=False)
    model_ecg.to(args.device)

    return model_eeg, model_ecg

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