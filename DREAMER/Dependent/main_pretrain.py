import pickle
import numpy as np
from tqdm import tqdm
import os
import random
import argparse
import torch
from torch.utils import data
from utils import data_aug
from train import pretrain
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='Pretrain the model for PysioSync')
parser.add_argument('--save-dir', default='./pretrain_pth',
                    help='weight_path')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run in pretrain')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', default=0.001, type=float)
parser.add_argument('--n-folds', default=10, type=int, metavar='N',
                    help='')
parser.add_argument('--B', default=8, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--N', default=2, type=int, metavar='N',
                    help='Number of subjects in minibatch')
parser.add_argument('--mlp-dim', default=512, type=int, metavar='N')
parser.add_argument('--embed-dim', default=128, type=int, metavar='N')
parser.add_argument('--eeg-dim', default=128 * 14, type=int, metavar='N')
parser.add_argument('--ecg-dim', default=128 * 2, type=int, metavar='N')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(42)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_eeg_path = './Data_processing/DREAMER_Pre/all_dataset_eeg.pkl'
    data_ecg_path = './Data_processing/DREAMER_Pre/all_dataset_ecg.pkl'
    label_path = './Data_processing/DREAMER_Pre/all_arousal_labels.pkl'
    with open(data_eeg_path, "rb") as fp:
        data_eeg = pickle.load(fp)
    with open(data_ecg_path, "rb") as fp:
        data_ecg = pickle.load(fp)
    with open(label_path, "rb") as fp:
        label = pickle.load(fp)
    index = list(range(len(label)))
    random.shuffle(index)
    data_eeg, data_ecg, label = data_eeg[index], data_ecg[index], label[index]

    folds_list = np.arange(0, args.n_folds)
    n_per = round(data_eeg.shape[0] / args.n_folds)

    results_pretrain = {}
    results_pretrain['best_epoch_eeg'], results_pretrain['best_epoch_ecg'] = np.zeros(args.n_folds), np.zeros(args.n_folds)

    for fold in folds_list:
        print('fold', fold)
        if fold < args.n_folds - 1:
            val_list = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_list = np.arange(n_per * fold, data_eeg.shape[0])
        val_list = [int(val) for val in val_list]
        print('val', val_list)
        train_list = np.array(list(set(np.arange(data_eeg.shape[0])) - set(val_list)))
        print('train', train_list)

        # Load training and test data
        x_train_eeg, x_train_ecg, y_train_eeg = [data_eeg[list(train_list)], data_ecg[list(train_list)],
                                                 label[list(train_list)]]
        x_test_eeg, x_test_ecg, y_test_eeg = [data_eeg[list(val_list)], data_ecg[list(val_list)],
                                                 label[list(val_list)]]
        # Print length of training labels
        print('label_train length', len(y_train_eeg))

        # # Data augmentation for EEG and ecg
        print('EEG data augmenting...')
        x_train_eeg, y_train_eeg = data_aug(x_train_eeg, y_train_eeg)
        
        print('ecg data augmenting...')
        x_train_ecg, _ = data_aug(x_train_ecg, y_train_eeg)

        # Convert to PyTorch tensors
        x_train_eeg, x_test_eeg = map(lambda x: torch.tensor(x, dtype=torch.float), [x_train_eeg, x_test_eeg])
        x_train_ecg, x_test_ecg = map(lambda x: torch.tensor(x, dtype=torch.float), [x_train_ecg, x_test_ecg])
        y_train, y_test = map(lambda x: torch.tensor(x, dtype=torch.long), [y_train_eeg, y_test_eeg])

        # Create DataLoader for training and testing
        train_loader = data.DataLoader(data.TensorDataset(x_train_eeg, x_train_ecg, y_train), args.B, shuffle=True)
        test_loader = data.DataLoader(data.TensorDataset(x_test_eeg, x_test_ecg, y_test), args.B, shuffle=True)

        # Set model storage address
        saved_models_dir = os.path.join(args.save_dir, str(fold) + '/')
        os.makedirs(saved_models_dir, exist_ok=True)
        if not os.path.exists(saved_models_dir):
            os.makedirs(saved_models_dir)

        print('Pretraining........')
        best_epoch_eeg, best_epoch_ecg = pretrain(args, train_loader, test_loader, saved_models_dir)

        results_pretrain['best_epoch_eeg'][fold] = best_epoch_eeg
        results_pretrain['best_epoch_ecg'][fold] = best_epoch_ecg

    with open(os.path.join(args.save_dir, 'best_epoch.pkl'), 'wb') as f:
        pickle.dump(results_pretrain, f)
