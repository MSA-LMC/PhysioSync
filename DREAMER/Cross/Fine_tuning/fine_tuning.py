import pickle
import random
import numpy as np
import os
import torch
import torch.optim as optim
import argparse
from models import classifier_max
from utils import load_models, prepare_data
from train import main
from models import Encoder_eeg, Encoder_ecg
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Pretrain the model for PysioSync')
parser.add_argument('--data-path', default='../Data_processing/DREAMER_Pre/',
                    help='data_path')
parser.add_argument('--pth-dir', default='../pretrain_pth',
                    help='weight_path')
parser.add_argument('--dimension', default='arousal',
                    help='arousal or valence or four')
parser.add_argument('--window_size', default=128, type=int,
                    help='data window')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run in pretrain')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', default=0.001, type=float)
parser.add_argument('--n-folds', default=23, type=int, metavar='N')
parser.add_argument('--B', default=256, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--N', default=2, type=int, metavar='N',
                    help='')
parser.add_argument('--mlp-dim', default=512, type=int, metavar='N')
parser.add_argument('--embed-dim', default=128, type=int, metavar='N')
parser.add_argument('--eeg-dim', default=128 * 14, type=int, metavar='N')
parser.add_argument('--ecg-dim', default=128 * 2, type=int, metavar='N')

if __name__ == '__main__':
    random.seed(42)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir_eeg = args.data_path + 'all_dataset_eeg.pkl'
    dataset_dir_ecg = args.data_path + 'all_dataset_ecg.pkl'
    dataset_dir_label = args.data_path + 'all_' + args.dimension + '_labels.pkl'
    epoch_path = args.pth_dir + '/best_epoch.pkl'
    ###load training set
    with open(dataset_dir_eeg, "rb") as fp:
        data_eeg = pickle.load(fp)
    with open(dataset_dir_ecg, "rb") as fp:
        data_ecg = pickle.load(fp)
    with open(dataset_dir_label, "rb") as fp:
        label = pickle.load(fp)
    with open(epoch_path, "rb") as fp:
        best_epoch = pickle.load(fp)

    index = list(range(len(label)))
    random.shuffle(index)
    data_eeg, data_ecg, label = data_eeg[index], data_ecg[index], label[index]

    data_eeg, data_ecg = (np.transpose(arr, (1, 0, 2, 3, 4)) for arr in [data_eeg, data_ecg])
    label = np.transpose(label, (1, 0))

    folds_list = np.arange(0, args.n_folds)
    n_per = round(data_eeg.shape[0] / args.n_folds)

    acc, f1 = [], []

    for fold in folds_list:
        print('fold', fold)
        if fold < args.n_folds - 1:
            val_list = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            # val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
            val_list = np.arange(n_per * fold, data_eeg.shape[0])
        val_list = [int(val) for val in val_list]
        print('val', val_list)

        train_list = np.array(list(set(np.arange(data_eeg.shape[0])) - set(val_list)))
        print('train', train_list)

        train_loader, test_loader = prepare_data(data_eeg, data_ecg, label, train_list, val_list, args)

        model_eeg, model_ecg = load_models(fold, best_epoch, args)
        # model_eeg = Encoder_eeg(embed_dim=args.embed_dim, eeg_dim=args.eeg_dim, mlp_dim=args.mlp_dim)
        # model_eeg.to(args.device)
        #
        # model_ecg = Encoder_ecg(embed_dim=args.embed_dim, ecg_dim=args.ecg_dim, mlp_dim=args.mlp_dim)
        # model_ecg.to(args.device)
        if args.dimension == 'arousal' or args.dimension == 'valence':
            args.num_classes = 2
        elif args.dimension == 'four':
            args.num_classes = 4

        model_classifier = classifier_max(num_classes=args.num_classes, mlp_dim=args.mlp_dim)
        model_classifier.to(args.device)

        all_parameters = list(model_eeg.parameters()) + list(model_ecg.parameters()) \
                         + list(model_classifier.parameters())

        optimizer = optim.Adam(all_parameters, lr=args.lr)

        # Training
        test_acc, test_f1 = main(model_ecg, model_eeg, model_classifier,
                                 optimizer, train_loader, test_loader, args)

        acc.append(test_acc)
        f1.append(test_f1)
    print('acc mean:', np.mean(acc))
    print('acc std:', np.std(acc))
    print('f1 mean:', np.mean(f1))
    print('f1 std:', np.std(f1))
