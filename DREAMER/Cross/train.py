import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import os
import time
from utils import save_checkpoint, info_nce_loss, accuracy
from models import Encoder_eeg, Encoder_ecg, CMCon, constrast_loss
torch.set_default_tensor_type(torch.FloatTensor)

def pretrain(args, train_loader, test_loader, saved_models_dir):
    # List of models
 
    models = [
        Encoder_eeg(embed_dim=args.embed_dim, eeg_dim=args.eeg_dim, mlp_dim=args.mlp_dim),
        Encoder_ecg(embed_dim=args.embed_dim, ecg_dim=args.ecg_dim, mlp_dim=args.mlp_dim),
        CMCon(mlp_dim=args.mlp_dim)
    ]

    model_eeg = models[0].to(args.device)
    model_ecg = models[1].to(args.device)
    model_cc = models[2].to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    all_parameters = list(model_eeg.parameters()) + list(model_ecg.parameters()) + list(model_cc.parameters())
    args.optimizer = optim.Adam(all_parameters, lr=args.lr, weight_decay=args.weight_decay)

    best_acc_eeg = 0
    best_epoch_eeg = 0
    best_acc_ecg = 0
    best_epoch_ecg = 0
    model_epoch_eeg, optimizer_epoch_eeg, model_epoch_ecg, optimizer_epoch_ecg = {}, {}, {}, {}
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_acces_tem_eeg, train_losses_tem_eeg, train_acces_tem_ecg, train_losses_tem_ecg = train(args,
            models,
            train_loader,
            criterion)

        print(f'Epoch: {epoch}')
        print(f'eeg_Train_loss: {train_losses_tem_eeg:.4f}, eeg_Train_acc: {train_acces_tem_eeg:.4f}')
        print(f'ecg_Train_loss: {train_losses_tem_ecg:.4f}, ecg_Train_acc: {train_acces_tem_ecg:.4f}')

        # 每个epoch测试一次
        test_acc_eeg, test_loss_eeg, test_acc_ecg, test_loss_ecg = eval(args,
            models,
            test_loader,
            criterion)

        model_epoch_eeg[epoch] = model_eeg
        optimizer_epoch_eeg[epoch] = args.optimizer

        model_epoch_ecg[epoch] = model_ecg
        optimizer_epoch_ecg[epoch] = args.optimizer

        print(f'Test_loss_eeg: {test_loss_eeg:.4f}, Test_acc_eeg: {test_acc_eeg:.4f}')
        print(f'Test_loss_ecg: {test_loss_ecg:.4f}, Test_acc_ecg: {test_acc_ecg:.4f}')

        if test_acc_eeg > best_acc_eeg:
            best_acc_eeg = test_acc_eeg
            best_epoch_eeg = epoch
        if test_acc_ecg > best_acc_ecg:
            best_acc_ecg = test_acc_ecg
            best_epoch_ecg = epoch
        end_time = time.time()
        print('time consumed:', end_time - start_time)
    print("EEG_Training has finished")
    print("eeg best epoch is:", best_epoch_eeg)
    print("ecg best epoch is:", best_epoch_ecg)

    best_model_eeg = model_epoch_eeg[best_epoch_eeg]
    best_optimizer_eeg = optimizer_epoch_eeg[best_epoch_eeg]
    checkpoint_name = 'eeg_best_checkpoint_{:04d}.pth'.format(best_epoch_eeg)
    save_checkpoint(
        {
            'epoch': best_epoch_eeg,
            'state_dict': best_model_eeg.state_dict(),
            'optimizer': best_optimizer_eeg.state_dict()
        }, is_best=False, filename=os.path.join(saved_models_dir, checkpoint_name)
    )
    best_model_ecg = model_epoch_ecg[best_epoch_eeg]
    best_optimizer_ecg = optimizer_epoch_ecg[best_epoch_ecg]
    checkpoint_name = 'ecg_best_checkpoint_{:04d}.pth'.format(best_epoch_ecg)
    save_checkpoint(
        {
            'epoch': best_epoch_ecg,
            'state_dict': best_model_ecg.state_dict(),
            'optimizer': best_optimizer_ecg.state_dict()
        }, is_best=False, filename=os.path.join(saved_models_dir, checkpoint_name)
    )
    return best_epoch_eeg, best_epoch_ecg


def train(args, models, train_loader, criterion):
    model_eeg = models[0].train()
    model_ecg = models[1].train()
    model_cc = models[2].train()

    iters = 0
    total_acc_eeg = 0
    total_acc_ecg = 0
    total_loss_eeg = 0
    total_loss_ecg = 0
    for eeg, ecg, y in tqdm(train_loader, desc='Training', unit='pairs'):
        iters += 1
        eeg = eeg.cpu().numpy()
        ecg = ecg.cpu().numpy()

        groups_eeg = torch.tensor(eeg, dtype=torch.float, device=args.device)
        groups_ecg = torch.tensor(ecg, dtype=torch.float, device=args.device)

        out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
        out_ecg, out_ecg_encoder = model_ecg(groups_ecg, mode='contrast')
        logits_eeg, labels_eeg = info_nce_loss(args, out_eeg)
        logits_ecg, labels_ecg = info_nce_loss(args, out_ecg)

        tem_loss_eeg = criterion(logits_eeg, labels_eeg)
        tem_loss_ecg = criterion(logits_ecg, labels_ecg)

        c_con_loss = model_cc(out_eeg_encoder, out_ecg_encoder, criterion, args)

        loss = 0.5 * tem_loss_eeg + 0.5 * tem_loss_ecg + c_con_loss

        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()

        top1_eeg, top5_eeg = accuracy(logits_eeg, labels_eeg, topk=(1, 5))
        top1_gsr, top5_gsr = accuracy(logits_ecg, labels_ecg, topk=(1, 5))

        total_acc_eeg += top1_eeg[0].cpu().numpy()
        total_acc_ecg += top1_gsr[0].cpu().numpy()
        total_loss_eeg += tem_loss_eeg
        total_loss_ecg += tem_loss_ecg
    train_acces_eeg = total_acc_eeg / iters
    train_acces_ecg = total_acc_ecg / iters
    train_losses_eeg = total_loss_eeg / iters
    train_losses_ecg = total_loss_ecg / iters

    return train_acces_eeg, train_losses_eeg, train_acces_ecg, train_losses_ecg


def eval(args, models, test_loader, criterion):
    model_eeg = models[0].eval()
    model_ecg = models[1].eval()
    model_cc = models[2].eval()

    iters = 0
    total_acc_eeg = 0
    total_acc_ecg = 0
    total_loss_eeg = 0
    total_loss_ecg = 0
    with torch.no_grad():
        for eeg, ecg, y in test_loader:
            iters += 1
            eeg = eeg.cpu().numpy()
            ecg = ecg.cpu().numpy()

            groups_eeg = torch.tensor(eeg, dtype=torch.float, device=args.device)
            groups_ecg = torch.tensor(ecg, dtype=torch.float, device=args.device)

            out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
            out_ecg, out_ecg_encoder = model_ecg(groups_ecg, mode='contrast')
            logits_eeg, labels_eeg = info_nce_loss(args, out_eeg)
            logits_ecg, labels_ecg = info_nce_loss(args, out_ecg)

            tem_loss_eeg = criterion(logits_eeg, labels_eeg)
            tem_loss_ecg = criterion(logits_ecg, labels_ecg)

            c_con_loss = model_cc(out_eeg_encoder, out_ecg_encoder, criterion, args)

            loss = 0.5 * tem_loss_eeg + 0.5 * tem_loss_ecg + c_con_loss

            top1_eeg, top5_eeg = accuracy(logits_eeg, labels_eeg, topk=(1, 5))
            top1_gsr, top5_gsr = accuracy(logits_ecg, labels_ecg, topk=(1, 5))

            total_acc_eeg += top1_eeg[0].cpu().numpy()
            total_acc_ecg += top1_gsr[0].cpu().numpy()
            total_loss_eeg += tem_loss_eeg
            total_loss_ecg += tem_loss_ecg
        test_acces_eeg = total_acc_eeg / iters
        test_acces_ecg = total_acc_ecg / iters
        test_losses_eeg = total_loss_eeg / iters
        test_losses_ecg = total_loss_ecg / iters
    return test_acces_eeg, test_losses_eeg, test_acces_ecg, test_losses_ecg
