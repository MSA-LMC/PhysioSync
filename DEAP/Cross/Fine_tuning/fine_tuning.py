import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.optim as optim
from utils import load_models, prepare_data
from train import main
from models import classifier_max
import argparse
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Pretrain the model for PysioSync')
parser.add_argument('--data-path', default='../Data_processing/DEAP_5s/',
                    help='data_path')
parser.add_argument('--save-dir', default='../pretrain_pth',
                    help='weight_path')
parser.add_argument('--dimension', default='A',
                    help='A or V or Four')
parser.add_argument('--window', default='5s',
                    help='data window')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run in pretrain')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', default=0.001, type=float)
parser.add_argument('--n-folds', default=10, type=int, metavar='N')
parser.add_argument('--B', default=256, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--N', default=2, type=int, metavar='N',
                    help='')
parser.add_argument('--mlp-dim_5s', default=2560, type=int, metavar='N')
parser.add_argument('--mlp-dim_1s', default=512, type=int, metavar='N')
parser.add_argument('--embed-dim-5s', default=640, type=int, metavar='N')
parser.add_argument('--embed-dim-1s', default=128, type=int, metavar='N')
parser.add_argument('--eeg-dim-5s', default=640*32, type=int, metavar='N')
parser.add_argument('--eeg-dim-1s', default=128*32, type=int, metavar='N')
parser.add_argument('--gsr-dim-5s', default=640, type=int, metavar='N')
parser.add_argument('--gsr-dim-1s', default=128, type=int, metavar='N')
if __name__ == '__main__':
    np.random.seed(42)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_eeg_path = args.data_path + args.dimension + '/DATA_DEAP.npy'
    data_gsr_path = args.data_path + 'GSR_' + args.dimension + '/DATA_DEAP.npy'
    label_path = args.data_path + args.dimension + '/LABEL_DEAP.npy'

    epoch_path_5s = args.save_dir + '_5s/best_epoch.pkl'
    root_dir_5s = args.save_dir + '_5s/'

    epoch_path_1s = args.save_dir + '_1s/best_epoch.pkl'
    root_dir_1s = args.save_dir + '_1s/'

    with open(epoch_path_5s, "rb") as fp:
        best_epoch = pickle.load(fp)
    with open(epoch_path_1s, "rb") as fp_1s:
        best_epoch_1s = pickle.load(fp_1s)
    data_eeg, data_gsr, label = [np.load(path) for path in [data_eeg_path, data_gsr_path, label_path]]
    index = np.random.permutation(len(label))
    data_eeg, data_gsr, label = data_eeg[index], data_gsr[index], label[index]

    data_eeg, data_gsr = (np.transpose(arr, (1, 0, 2, 3, 4)) for arr in [data_eeg, data_gsr])
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

        train_loader, test_loader = prepare_data(data_eeg, data_gsr, label, train_list, val_list, args)

        model_eeg, model_eeg_1s, model_gsr, model_gsr_1s = load_models(
            fold, best_epoch, best_epoch_1s, root_dir_5s, root_dir_1s, args)
        if args.dimension == 'A' or args.dimension == 'V':
            args.num_classes = 2
        elif args.dimension == 'Four':
            args.num_classes = 4
        model_classifier = classifier_max(num_classes=args.num_classes)
        model_classifier.to(args.device)

        all_parameters = list(model_eeg.parameters()) + list(model_gsr.parameters()) \
                         + list(model_classifier.parameters()) + list(model_eeg_1s.parameters()) + list(model_gsr_1s.parameters())

        optimizer = optim.Adam(all_parameters, lr=args.lr)

        # Training
        test_acc, test_f1 = main(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s,
                                       model_classifier, optimizer, train_loader, test_loader, args)

        acc.append(test_acc)
        f1.append(test_f1)
    print('acc mean:', np.mean(acc))
    print('acc std:', np.std(acc))
    print('f1 mean:', np.mean(f1))
    print('f1 std:', np.std(f1))
