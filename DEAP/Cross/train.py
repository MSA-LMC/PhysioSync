import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
import os
import time
from utils import save_checkpoint, info_nce_loss, accuracy
from models import Encoder_eeg, Encoder_gsr, CMCon, constrast_loss
torch.set_default_tensor_type(torch.FloatTensor)
def pretrain(args, train_loader, test_loader, saved_models_dir):
    # List of models
    if args.window == '5s':
        models = [
            Encoder_eeg(embed_dim=args.embed_dim_5s, eeg_dim=args.eeg_dim_5s, mlp_dim=args.mlp_dim_5s),
            Encoder_gsr(embed_dim=args.embed_dim_5s, gsr_dim=args.gsr_dim_5s, mlp_dim=args.mlp_dim_5s),
            CMCon(mlp_dim=args.mlp_dim_5s)
        ]
    elif args.window == '1s':
        models = [
            Encoder_eeg(embed_dim=args.embed_dim_1s, eeg_dim=args.eeg_dim_1s, mlp_dim=args.mlp_dim_1s),
            Encoder_gsr(embed_dim=args.embed_dim_1s, gsr_dim=args.gsr_dim_1s, mlp_dim=args.mlp_dim_1s),
            CMCon(mlp_dim=args.mlp_dim_1s)
        ]

    model_eeg = models[0].to(args.device)
    model_gsr = models[1].to(args.device)
    model_cc = models[2].to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    all_parameters = list(model_eeg.parameters()) + list(model_gsr.parameters()) + list(model_cc.parameters())
    args.optimizer = optim.Adam(all_parameters, lr=args.lr, weight_decay=args.weight_decay)

    best_acc_eeg = 0
    best_epoch_eeg = 0
    best_acc_gsr = 0
    best_epoch_gsr = 0
    model_epoch_eeg, optimizer_epoch_eeg, model_epoch_gsr, optimizer_epoch_gsr = {}, {}, {}, {}
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_acces_tem_eeg, train_losses_tem_eeg, train_acces_tem_gsr, train_losses_tem_gsr = train(args,
            model_eeg, model_gsr, model_cc,
            train_loader,
            criterion)

        print(f'Epoch: {epoch}')
        print(f'eeg_Train_loss: {train_losses_tem_eeg:.4f}, eeg_Train_acc: {train_acces_tem_eeg:.4f}')
        print(f'gsr_Train_loss: {train_losses_tem_gsr:.4f}, gsr_Train_acc: {train_acces_tem_gsr:.4f}')

        # 每个epoch测试一次
        test_acc_eeg, test_loss_eeg, test_acc_gsr, test_loss_gsr = eval(args,
            model_eeg, model_gsr, model_cc,
            test_loader,
            criterion)

        model_epoch_eeg[epoch] = model_eeg
        optimizer_epoch_eeg[epoch] = args.optimizer

        model_epoch_gsr[epoch] = model_gsr
        optimizer_epoch_gsr[epoch] = args.optimizer

        print(f'Test_loss_eeg: {test_loss_eeg:.4f}, Test_acc_eeg: {test_acc_eeg:.4f}')
        print(f'Test_loss_gsr: {test_loss_gsr:.4f}, Test_acc_gsr: {test_acc_gsr:.4f}')

        if test_acc_eeg > best_acc_eeg:
            best_acc_eeg = test_acc_eeg
            best_epoch_eeg = epoch
        if test_acc_gsr > best_acc_gsr:
            best_acc_gsr = test_acc_gsr
            best_epoch_gsr = epoch
        end_time = time.time()
        print('time consumed:', end_time - start_time)
    print("EEG_Training has finished")
    print("eeg best epoch is:", best_epoch_eeg)
    print("gsr best epoch is:", best_epoch_gsr)

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
    best_model_gsr = model_epoch_gsr[best_epoch_eeg]
    best_optimizer_gsr = optimizer_epoch_gsr[best_epoch_gsr]
    checkpoint_name = 'gsr_best_checkpoint_{:04d}.pth'.format(best_epoch_gsr)
    save_checkpoint(
        {
            'epoch': best_epoch_gsr,
            'state_dict': best_model_gsr.state_dict(),
            'optimizer': best_optimizer_gsr.state_dict()
        }, is_best=False, filename=os.path.join(saved_models_dir, checkpoint_name)
    )
    return best_epoch_eeg, best_epoch_gsr


def train(args, model_eeg, model_gsr, model_cc, train_loader, criterion):
    model_eeg.train()
    model_gsr.train()
    model_cc.train()

    iters = 0
    total_acc_eeg = 0
    total_acc_gsr = 0
    total_loss_eeg = 0
    total_loss_gsr = 0
    for eeg, gsr, y in tqdm(train_loader, desc='Training', unit='pairs'):
        iters += 1
        eeg = eeg.cpu().numpy()
        gsr = gsr.cpu().numpy()

        groups_eeg = torch.tensor(eeg, dtype=torch.float, device=args.device)

        groups_gsr = torch.tensor(gsr, dtype=torch.float, device=args.device)

        out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
        out_gsr, out_gsr_encoder = model_gsr(groups_gsr, mode='contrast')

        logits_eeg, labels_eeg = info_nce_loss(args, out_eeg)
        logits_gsr, labels_gsr = info_nce_loss(args, out_gsr)

        tem_loss_eeg = criterion(logits_eeg, labels_eeg)
        tem_loss_gsr = criterion(logits_gsr, labels_gsr)

        c_con_loss = model_cc(out_eeg_encoder, out_gsr_encoder, criterion, args)

        loss = 0.5 * tem_loss_eeg + 0.5 * tem_loss_gsr + c_con_loss

        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()

        top1_eeg, top5_eeg = accuracy(logits_eeg, labels_eeg, topk=(1, 5))
        top1_gsr, top5_gsr = accuracy(logits_gsr, labels_gsr, topk=(1, 5))

        total_acc_eeg += top1_eeg[0].cpu().numpy()
        total_acc_gsr += top1_gsr[0].cpu().numpy()
        total_loss_eeg += tem_loss_eeg
        total_loss_gsr += tem_loss_gsr
    train_acces_eeg = total_acc_eeg / iters
    train_acces_gsr = total_acc_gsr / iters
    train_losses_eeg = total_loss_eeg / iters
    train_losses_gsr = total_loss_gsr / iters

    return train_acces_eeg, train_losses_eeg, train_acces_gsr, train_losses_gsr


def eval(args, model_eeg, model_gsr, model_cc, test_loader, criterion):
    model_eeg.eval()
    model_gsr.eval()
    model_cc.eval()

    iters = 0
    total_acc_eeg = 0
    total_acc_gsr = 0
    total_loss_eeg = 0
    total_loss_gsr = 0
    with torch.no_grad():
        for eeg, gsr, y in test_loader:
            iters += 1
            eeg = eeg.cpu().numpy()
            gsr = gsr.cpu().numpy()

            groups_eeg = torch.tensor(eeg, dtype=torch.float, device=args.device)

            groups_gsr = torch.tensor(gsr, dtype=torch.float, device=args.device)

            out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
            out_gsr, out_gsr_encoder = model_gsr(groups_gsr, mode='contrast')

            logits_eeg, labels_eeg = info_nce_loss(args, out_eeg)
            logits_gsr, labels_gsr = info_nce_loss(args, out_gsr)

            tem_loss_eeg = criterion(logits_eeg, labels_eeg)
            tem_loss_gsr = criterion(logits_gsr, labels_gsr)

            c_con_loss = model_cc(out_eeg_encoder, out_gsr_encoder, criterion, args)

            loss = 0.5 * tem_loss_eeg + 0.5 * tem_loss_gsr + c_con_loss

            top1_eeg, top5_eeg = accuracy(logits_eeg, labels_eeg, topk=(1, 5))
            top1_gsr, top5_gsr = accuracy(logits_gsr, labels_gsr, topk=(1, 5))

            total_acc_eeg += top1_eeg[0].cpu().numpy()
            total_acc_gsr += top1_gsr[0].cpu().numpy()
            total_loss_eeg += tem_loss_eeg
            total_loss_gsr += tem_loss_gsr
        test_acces_eeg = total_acc_eeg / iters
        test_acces_gsr = total_acc_gsr / iters
        test_losses_eeg = total_loss_eeg / iters
        test_losses_gsr = total_loss_gsr / iters
    return test_acces_eeg, test_losses_eeg, test_acces_gsr, test_losses_gsr
