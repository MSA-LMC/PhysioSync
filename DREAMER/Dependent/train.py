import numpy as np
import torch
from torch import nn
import torch.optim as optim
import os
import time
from utils import save_checkpoint
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
    for eeg, ecg, y in train_loader:
        iters += 1
        bs,ss = eeg.shape[0], eeg.shape[1]
        eeg = eeg.cpu().numpy()
        ecg = ecg.cpu().numpy()

        groups_1_eeg, groups_1_ecg = [], []
        groups_2_eeg, groups_2_ecg = [], []

        for i in range(bs):
            rand_subs1 = np.random.randint(0, ss / 2, size=args.N)
            rand_subs2 = np.random.randint(ss / 2, ss, size=args.N)
            groups_1_eeg.append(eeg[i, rand_subs1])
            groups_1_ecg.append(ecg[i, rand_subs1])
            groups_2_eeg.append(eeg[i, rand_subs2])
            groups_2_ecg.append(ecg[i, rand_subs2])

        groups_eeg = groups_1_eeg + groups_2_eeg
        groups_ecg = groups_1_ecg + groups_2_ecg

        groups_eeg = np.concatenate(groups_eeg)
        groups_eeg = groups_eeg.reshape(-1, groups_eeg.shape[-3],groups_eeg.shape[-2],groups_eeg.shape[-1])
        groups_eeg = torch.tensor(groups_eeg, dtype=torch.float, device=args.device)

        groups_ecg = np.concatenate(groups_ecg)
        groups_ecg = groups_ecg.reshape(-1, groups_ecg.shape[-3], groups_ecg.shape[-2], groups_ecg.shape[-1])
        groups_ecg = torch.tensor(groups_ecg, dtype=torch.float, device=args.device)

        out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
        out_ecg, out_ecg_encoder = model_ecg(groups_ecg, mode='contrast')
        if args.N > 1 :
            out_eeg = torch.reshape(out_eeg, (-1, args.N, out_eeg.shape[-1]))
            out_eeg = torch.max(out_eeg, dim=1)[0]
            out_ecg = torch.reshape(out_ecg, (-1, args.N, out_ecg.shape[-1]))
            out_ecg = torch.max(out_ecg, dim=1)[0]

            out_eeg_encoder = torch.reshape(out_eeg_encoder, (-1, args.N, out_eeg_encoder.shape[-1]))
            out_eeg_encoder = torch.max(out_eeg_encoder, dim=1)[0]
            out_ecg_encoder = torch.reshape(out_ecg_encoder, (-1, args.N, out_ecg_encoder.shape[-1]))
            out_ecg_encoder = torch.max(out_ecg_encoder, dim=1)[0]

        c_con_loss = model_cc(out_eeg_encoder, out_ecg_encoder, criterion, args.tau)

        args.optimizer.zero_grad()
        tem_loss_eeg, lab_con_eeg, logits_ab_eeg = constrast_loss(out_eeg, criterion, args.tau)
        _, log_p_eeg = torch.max(logits_ab_eeg.data, 1)
        tem_loss_ecg, lab_con_ecg, logits_ab_ecg = constrast_loss(out_ecg, criterion, args.tau)
        _, log_p_ecg = torch.max(logits_ab_ecg.data, 1)

        loss = 0.5 * tem_loss_eeg + 0.5 * tem_loss_ecg + c_con_loss
        evaluation_batch_eeg = ((log_p_eeg == lab_con_eeg).cpu().numpy()*1)
        evaluation_batch_ecg = ((log_p_ecg == lab_con_ecg).cpu().numpy()*1)

        loss.backward()
        args.optimizer.step()
        acc_eeg = sum(evaluation_batch_eeg) / evaluation_batch_eeg.shape[0]
        acc_ecg = sum(evaluation_batch_ecg) / evaluation_batch_ecg.shape[0]
        total_acc_eeg += acc_eeg
        total_acc_ecg += acc_ecg
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
            bs, ss = eeg.shape[0], eeg.shape[1]
            eeg = eeg.cpu().numpy()
            ecg = ecg.cpu().numpy()

            groups_1_eeg, groups_1_ecg = [], []
            groups_2_eeg, groups_2_ecg = [], []

            for i in range(bs):
                rand_subs1 = np.random.randint(0, ss / 2, size=args.N)
                rand_subs2 = np.random.randint(ss / 2, ss, size=args.N)
                groups_1_eeg.append(eeg[i, rand_subs1])
                groups_1_ecg.append(ecg[i, rand_subs1])
                groups_2_eeg.append(eeg[i, rand_subs2])
                groups_2_ecg.append(ecg[i, rand_subs2])

            groups_eeg = groups_1_eeg + groups_2_eeg
            groups_ecg = groups_1_ecg + groups_2_ecg

            groups_eeg = np.concatenate(groups_eeg)
            groups_eeg = groups_eeg.reshape(-1, groups_eeg.shape[-3], groups_eeg.shape[-2], groups_eeg.shape[-1])
            groups_eeg = torch.tensor(groups_eeg, dtype=torch.float, device=args.device)

            groups_ecg = np.concatenate(groups_ecg)
            groups_ecg = groups_ecg.reshape(-1, groups_ecg.shape[-3], groups_ecg.shape[-2], groups_ecg.shape[-1])
            groups_ecg = torch.tensor(groups_ecg, dtype=torch.float, device=args.device)

            out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
            out_ecg, out_ecg_encoder = model_ecg(groups_ecg, mode='contrast')
            if args.N > 1:
                out_eeg = torch.reshape(out_eeg, (-1, args.N, out_eeg.shape[-1]))
                out_eeg = torch.max(out_eeg, dim=1)[0]
                out_ecg = torch.reshape(out_ecg, (-1, args.N, out_ecg.shape[-1]))
                out_ecg = torch.max(out_ecg, dim=1)[0]

                out_eeg_encoder = torch.reshape(out_eeg_encoder, (-1, args.N, out_eeg_encoder.shape[-1]))
                out_eeg_encoder = torch.max(out_eeg_encoder, dim=1)[0]
                out_ecg_encoder = torch.reshape(out_ecg_encoder, (-1, args.N, out_ecg_encoder.shape[-1]))
                out_ecg_encoder = torch.max(out_ecg_encoder, dim=1)[0]
            c_con_loss = model_cc(out_eeg_encoder, out_ecg_encoder, criterion, args.tau)

            tem_loss_eeg, lab_con_eeg, logits_ab_eeg = constrast_loss(out_eeg, criterion, args.tau)
            _, log_p_eeg = torch.max(logits_ab_eeg.data, 1)
            tem_loss_ecg, lab_con_ecg, logits_ab_ecg = constrast_loss(out_ecg, criterion, args.tau)
            _, log_p_ecg = torch.max(logits_ab_ecg.data, 1)

            # loss = (0.5 * tem_loss_eeg) + (0.5 * tem_loss_ecg) + c_con_loss

            evaluation_batch_eeg = ((log_p_eeg == lab_con_eeg).cpu().numpy() * 1)
            evaluation_batch_ecg = ((log_p_ecg == lab_con_ecg).cpu().numpy() * 1)
            acc_eeg = sum(evaluation_batch_eeg) / evaluation_batch_eeg.shape[0]
            acc_ecg = sum(evaluation_batch_ecg) / evaluation_batch_ecg.shape[0]
            total_acc_eeg += acc_eeg
            total_acc_ecg += acc_ecg
            total_loss_eeg += tem_loss_eeg
            total_loss_ecg += tem_loss_ecg
        test_acces_eeg = total_acc_eeg / iters
        test_acces_ecg = total_acc_ecg / iters
        test_losses_eeg = total_loss_eeg / iters
        test_losses_ecg = total_loss_ecg / iters
    return test_acces_eeg, test_losses_eeg, test_acces_ecg, test_losses_ecg
