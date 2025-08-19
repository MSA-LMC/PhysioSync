import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from utils import matrix_percent
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings

def main(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s, model_classifier,
         optimizer, train_loader, test_loader, args):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 3,
                                                                     eta_min=0,
                                                                     last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss().to(args.device)
    train_losses = []
    train_acces = []
    test_loss, test_acc, cm, test_f1 = eval(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s,
                                            model_classifier, test_loader, loss_fn, args)
    print(f'Epoch {0}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}, Test f1 {test_f1:.4f}')
    print(f'confusion_matrix:{cm}')

    for epoch in range(1, args.epochs+1):
        model_classifier.train()
        train_losses_tem, train_acces_tem = train(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s,
                                                  model_classifier, train_loader, optimizer, loss_fn)
        train_acces.append(train_acces_tem)
        train_losses.append(train_losses_tem)
        print(f'Epoch {epoch}, Train loss {train_losses_tem:.4f}, Train acc {train_acces_tem:.4f}')
        scheduler.step()
        print('learning rate', scheduler.get_lr()[0])

        test_loss, test_acc, cm, test_f1 = eval(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s,
                                                model_classifier, test_loader, loss_fn, args)
        print(f'Epoch {epoch}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}, Test f1 {test_f1:.4f}')
        print(f'confusion_matrix:{cm}')

    test_loss, test_acc, cm, test_f1 = eval(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s,
                                            model_classifier, test_loader, loss_fn, args)
    print(f'Test loss {test_loss:.4f}, Test acc {test_acc:.4f}, Test f1 {test_f1:.4f}')
    print(f'confusion_matrix:{cm}')

    return test_acc, test_f1

def eval(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s,
                                            model_classifier, test_loader, loss_fn, args):
    model_gsr.eval()
    model_eeg.eval()
    model_gsr_1s.eval()
    model_eeg_1s.eval()
    model_classifier.eval()
    total_cm, total_loss, total_acc, total_f1, iter = 0,0,0,0,0
    with torch.no_grad():
        for pair in test_loader:
            x_eeg, x_gsr, y = pair[0], pair[1], pair[2]
            slices = [(0, 128), (128, 256), (256, 384), (384, 512), (512, None)]
            x_eeg_list = []
            x_gsr_list = []

            for start, end in slices:
                x_eeg_list.append(x_eeg[:, :, :, start:end])
                x_gsr_list.append(x_gsr[:, :, :, start:end])

            x_eeg = x_eeg.cuda().float().contiguous()
            x_gsr = x_gsr.cuda().float().contiguous()
            x_eeg_list = [x.cuda().float().contiguous() for x in x_eeg_list]
            x_gsr_list = [x.cuda().float().contiguous() for x in x_gsr_list]
            y = y.cuda().long().contiguous()

            out_eeg = model_eeg(x_eeg, mode='embedding')
            out_gsr = model_gsr(x_gsr, mode='embedding')
            out_eeg_list = [model_eeg_1s(x, mode='embedding') for x in x_eeg_list]
            out_gsr_list = [model_gsr_1s(x, mode='embedding') for x in x_gsr_list]

            out = model_classifier(out_eeg, out_gsr, *out_eeg_list, *out_gsr_list)

            y_pred = torch.argmax(out, 1)
            acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float()/y.shape[0]).item()
            loss = loss_fn(out, y)

            loss = loss.item()
            total_loss += loss
            total_acc += acc
            iter += 1
            y_true = y.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            cm = matrix_percent(y_true, y_pred)
            if args.dimension == "A" or args.dimension == "V":
                f1_sco = f1_score(y_true, y_pred)
            elif args.dimension == "Four":
                f1_sco = f1_score(y_true, y_pred, average='weighted') # dimension == "Four"

            total_f1 += f1_sco
            total_cm += cm
    test_loss = total_loss / iter
    test_acc = total_acc / iter
    f_cm = total_cm/iter
    test_f1 = total_f1/iter
    return test_loss, test_acc, f_cm, test_f1

#Training function
def train(model_gsr, model_eeg, model_eeg_1s, model_gsr_1s,
                                        model_classifier, train_loader, optimizer, loss_fn):
    model_gsr.train()
    model_eeg.train()
    model_gsr_1s.train()
    model_eeg_1s.train()
    model_classifier.train()
    train_losses_tem = []
    train_acces_tem = []
    i = 0
    for pair in tqdm(train_loader, desc='Processing', unit='pairs'):
        x_eeg, x_gsr, y = pair[0], pair[1], pair[2]
        slices = [(0, 128), (128, 256), (256, 384), (384, 512), (512, None)]
        x_eeg_list = []
        x_gsr_list = []

        for start, end in slices:
            x_eeg_list.append(x_eeg[:, :, :, start:end])
            x_gsr_list.append(x_gsr[:, :, :, start:end])

        x_eeg = x_eeg.cuda().float().contiguous()
        x_gsr = x_gsr.cuda().float().contiguous()
        x_eeg_list = [x.cuda().float().contiguous() for x in x_eeg_list]
        x_gsr_list = [x.cuda().float().contiguous() for x in x_gsr_list]
        y = y.cuda().long().contiguous()

        out_eeg = model_eeg(x_eeg, mode='embedding')
        out_gsr = model_gsr(x_gsr, mode='embedding')
        out_eeg_list = [model_eeg_1s(x, mode='embedding') for x in x_eeg_list]
        out_gsr_list = [model_gsr_1s(x, mode='embedding') for x in x_gsr_list]

        out = model_classifier(out_eeg, out_gsr, *out_eeg_list, *out_gsr_list)
        acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float() / y.shape[0]).item()

        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        optimizer.step()
        train_losses_tem.append(loss)
        train_acces_tem.append(acc)
        i += 1
    train_losses_tem = sum(train_losses_tem) / len(train_losses_tem)
    train_acces_tem = sum(train_acces_tem) / len(train_acces_tem)
    return train_losses_tem, train_acces_tem
