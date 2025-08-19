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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Pretrain the model for PysioSync')
parser.add_argument('--data-path', default='../Data_processing/',
                    help='data_path')
parser.add_argument('--save-dir', default='../pretrain_pth_',
                    help='weight_path')
parser.add_argument('--dimension', default='A',
                    help='A or V or Four, You can choose any one during the pre-training phase')
parser.add_argument('--stratified', default='no',
                    help='')
parser.add_argument('--window', default='5s',
                    help='data window: 1s or 5s')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run in pretrain')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature')
parser.add_argument('--lr', default=0.0007, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', default=0.001, type=float)
parser.add_argument('--n-folds', default=32, type=int, metavar='N',
                    help='')
parser.add_argument('--B', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--N', default=2, type=int, metavar='N',
                    help='Number of subjects in minibatch')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help=' n views in contrastive learning')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--mlp-dim-5s', default=2560, type=int, metavar='N')
parser.add_argument('--embed-dim-5s', default=640, type=int, metavar='N')
parser.add_argument('--eeg-dim-5s', default=640 * 32, type=int, metavar='N')
parser.add_argument('--gsr-dim-5s', default=640, type=int, metavar='N')
parser.add_argument('--mlp-dim-1s', default=512, type=int, metavar='N')
parser.add_argument('--embed-dim-1s', default=128, type=int, metavar='N')
parser.add_argument('--eeg-dim-1s', default=128 * 32, type=int, metavar='N')
parser.add_argument('--gsr-dim-1s', default=128, type=int, metavar='N')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(42)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_eeg_path = args.data_path + 'DEAP_' + args.window + '/' + args.dimension + '/DATA_DEAP.npy'
    data_gsr_path = args.data_path + 'DEAP_' + args.window + '/' + 'GSR_' + args.dimension + '/DATA_DEAP.npy'
    label_path = args.data_path + 'DEAP_' + args.window + '/' + args.dimension + '/LABEL_DEAP.npy'
    data_eeg, data_gsr, label = [np.load(path) for path in [data_eeg_path, data_gsr_path, label_path]]
    index = list(range(len(label)))
    random.shuffle(index)
    data_eeg, data_gsr, label = data_eeg[index], data_gsr[index], label[index]
    data_eeg, data_gsr = (np.transpose(arr, (1, 0, 2, 3, 4)) for arr in [data_eeg, data_gsr])
    label = np.transpose(label, (1, 0))

    folds_list = np.arange(0, args.n_folds)
    n_per = round(data_eeg.shape[0] / args.n_folds)

    results_pretrain = {}
    results_pretrain['best_epoch_eeg'], results_pretrain['best_epoch_gsr'] = np.zeros(args.n_folds), np.zeros(args.n_folds)

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
        x_train_eeg, x_train_gsr, y_train_eeg = [data_eeg[list(train_list)], data_gsr[list(train_list)],
                                                 label[list(train_list)]]
        x_test_eeg, x_test_gsr, y_test_eeg = [data_eeg[list(val_list)], data_gsr[list(val_list)],
                                                 label[list(val_list)]]

        # x_train_eeg, x_train_gsr, x_test_eeg, x_test_gsr = \
        #     (np.transpose(arr, (1, 0, 2, 3, 4)) for arr in [x_train_eeg, x_train_gsr, x_test_eeg, x_test_gsr])
        # y_train_eeg, y_test_eeg = (np.transpose(arr, (1, 0)) for arr in [y_train_eeg, y_test_eeg])

        # Print length of training labels
        print('label_train length', len(y_train_eeg))

        # # Data augmentation for EEG and GSR
        print('EEG data augmenting...')
        x_train_eeg, y_train_eeg = data_aug(x_train_eeg, y_train_eeg)

        print('GSR data augmenting...')
        x_train_gsr, _ = data_aug(x_train_gsr, y_train_eeg)

        x_train_eeg = x_train_eeg.reshape(-1, 1, x_train_eeg.shape[-2], x_train_eeg.shape[-1])
        x_train_gsr = x_train_gsr.reshape(-1, 1, x_train_gsr.shape[-2], x_train_gsr.shape[-1])
        x_test_eeg = x_test_eeg.reshape(-1, 1, x_test_eeg.shape[-2], x_test_eeg.shape[-1])
        x_test_gsr = x_test_gsr.reshape(-1, 1, x_test_gsr.shape[-2], x_test_gsr.shape[-1])
        y_train_eeg = y_train_eeg.reshape(-1, )
        y_test_eeg = y_test_eeg.reshape(-1, )

        # Convert to PyTorch tensors
        x_train_eeg, x_test_eeg = map(lambda x: torch.tensor(x, dtype=torch.float), [x_train_eeg, x_test_eeg])
        x_train_gsr, x_test_gsr = map(lambda x: torch.tensor(x, dtype=torch.float), [x_train_gsr, x_test_gsr])
        y_train, y_test = map(lambda x: torch.tensor(x, dtype=torch.long), [y_train_eeg, y_test_eeg])

        # Create DataLoader for training and testing
        train_loader = data.DataLoader(data.TensorDataset(x_train_eeg, x_train_gsr, y_train), args.B, shuffle=True)
        test_loader = data.DataLoader(data.TensorDataset(x_test_eeg, x_test_gsr, y_test), args.B, shuffle=True)

        # Set model storage address
        saved_models_dir = os.path.join(args.save_dir + args.window, str(fold) + '/')
        os.makedirs(saved_models_dir, exist_ok=True)
        if not os.path.exists(saved_models_dir):
            os.makedirs(saved_models_dir)

        print('Pretraining........')
        best_epoch_eeg, best_epoch_gsr = pretrain(args, train_loader, test_loader, saved_models_dir)

        results_pretrain['best_epoch_eeg'][fold] = best_epoch_eeg
        results_pretrain['best_epoch_gsr'][fold] = best_epoch_gsr

    with open(os.path.join(args.save_dir + args.window, 'best_epoch.pkl'), 'wb') as f:
        pickle.dump(results_pretrain, f)
