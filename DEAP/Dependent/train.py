import numpy as np
import torch
from torch import nn
import torch.optim as optim
import os
import time
from utils import save_checkpoint
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
            models,
            train_loader,
            criterion)

        print(f'Epoch: {epoch}')
        print(f'eeg_Train_loss: {train_losses_tem_eeg:.4f}, eeg_Train_acc: {train_acces_tem_eeg:.4f}')
        print(f'gsr_Train_loss: {train_losses_tem_gsr:.4f}, gsr_Train_acc: {train_acces_tem_gsr:.4f}')

        # 每个epoch测试一次
        test_acc_eeg, test_loss_eeg, test_acc_gsr, test_loss_gsr = eval(args,
            models,
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


def train(args, models, train_loader, criterion):
    model_eeg = models[0].train()
    model_gsr = models[1].train()
    model_cc = models[2].train()

    iters = 0
    total_acc_eeg = 0
    total_acc_gsr = 0
    total_loss_eeg = 0
    total_loss_gsr = 0
    for eeg, gsr, y in train_loader:
        iters += 1
        bs,ss = eeg.shape[0], eeg.shape[1]
        eeg = eeg.cpu().numpy()
        gsr = gsr.cpu().numpy()

        groups_1_eeg, groups_1_gsr = [], []
        groups_2_eeg, groups_2_gsr = [], []

        for i in range(bs):
            rand_subs1 = np.random.randint(0, ss / 2, size=args.N)
            rand_subs2 = np.random.randint(ss / 2, ss, size=args.N)
            groups_1_eeg.append(eeg[i, rand_subs1])
            groups_1_gsr.append(gsr[i, rand_subs1])
            groups_2_eeg.append(eeg[i, rand_subs2])
            groups_2_gsr.append(gsr[i, rand_subs2])

        groups_eeg = groups_1_eeg + groups_2_eeg
        groups_gsr = groups_1_gsr + groups_2_gsr

        groups_eeg = np.concatenate(groups_eeg)
        groups_eeg = groups_eeg.reshape(-1, groups_eeg.shape[-3],groups_eeg.shape[-2],groups_eeg.shape[-1])
        groups_eeg = torch.tensor(groups_eeg, dtype=torch.float, device=args.device)

        groups_gsr = np.concatenate(groups_gsr)
        groups_gsr = groups_gsr.reshape(-1, groups_gsr.shape[-3], groups_gsr.shape[-2], groups_gsr.shape[-1])
        groups_gsr = torch.tensor(groups_gsr, dtype=torch.float, device=args.device)

        out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
        out_gsr, out_gsr_encoder = model_gsr(groups_gsr, mode='contrast')
        if args.N > 1 :
            out_eeg = torch.reshape(out_eeg, (-1, args.N, out_eeg.shape[-1]))
            out_eeg = torch.max(out_eeg, dim=1)[0]
            out_gsr = torch.reshape(out_gsr, (-1, args.N, out_gsr.shape[-1]))
            out_gsr = torch.max(out_gsr, dim=1)[0]

            out_eeg_encoder = torch.reshape(out_eeg_encoder, (-1, args.N, out_eeg_encoder.shape[-1]))
            out_eeg_encoder = torch.max(out_eeg_encoder, dim=1)[0]
            out_gsr_encoder = torch.reshape(out_gsr_encoder, (-1, args.N, out_gsr_encoder.shape[-1]))
            out_gsr_encoder = torch.max(out_gsr_encoder, dim=1)[0]

        c_con_loss = model_cc(out_eeg_encoder, out_gsr_encoder, criterion, args.tau)

        args.optimizer.zero_grad()
        tem_loss_eeg, lab_con_eeg, logits_ab_eeg = constrast_loss(out_eeg, criterion, args.tau)
        _, log_p_eeg = torch.max(logits_ab_eeg.data, 1)
        tem_loss_gsr, lab_con_gsr, logits_ab_gsr = constrast_loss(out_gsr, criterion, args.tau)
        _, log_p_gsr = torch.max(logits_ab_gsr.data, 1)

        loss = 0.5 * tem_loss_eeg + 0.5 * tem_loss_gsr + c_con_loss
        evaluation_batch_eeg = ((log_p_eeg == lab_con_eeg).cpu().numpy()*1)
        evaluation_batch_gsr = ((log_p_gsr == lab_con_gsr).cpu().numpy()*1)

        loss.backward()
        loss = loss.item()
        args.optimizer.step()
        acc_eeg = sum(evaluation_batch_eeg) / evaluation_batch_eeg.shape[0]
        acc_gsr = sum(evaluation_batch_gsr) / evaluation_batch_gsr.shape[0]
        total_acc_eeg += acc_eeg
        total_acc_gsr += acc_gsr
        total_loss_eeg += tem_loss_eeg
        total_loss_gsr += tem_loss_gsr
    train_acces_eeg = total_acc_eeg / iters
    train_acces_gsr = total_acc_gsr / iters
    train_losses_eeg = total_loss_eeg / iters
    train_losses_gsr = total_loss_gsr / iters

    return train_acces_eeg, train_losses_eeg, train_acces_gsr, train_losses_gsr


def eval(args, models, test_loader, criterion):
    model_eeg = models[0].eval()
    model_gsr = models[1].eval()
    model_cc = models[2].eval()

    iters = 0
    total_acc_eeg = 0
    total_acc_gsr = 0
    total_loss_eeg = 0
    total_loss_gsr = 0
    with torch.no_grad():
        for eeg, gsr, y in test_loader:
            iters += 1
            bs, ss = eeg.shape[0], eeg.shape[1]
            eeg = eeg.cpu().numpy()
            gsr = gsr.cpu().numpy()

            groups_1_eeg, groups_1_gsr = [], []
            groups_2_eeg, groups_2_gsr = [], []

            for i in range(bs):
                rand_subs1 = np.random.randint(0, ss / 2, size=args.N)
                rand_subs2 = np.random.randint(ss / 2, ss, size=args.N)
                groups_1_eeg.append(eeg[i, rand_subs1])
                groups_1_gsr.append(gsr[i, rand_subs1])
                groups_2_eeg.append(eeg[i, rand_subs2])
                groups_2_gsr.append(gsr[i, rand_subs2])

            groups_eeg = groups_1_eeg + groups_2_eeg
            groups_gsr = groups_1_gsr + groups_2_gsr

            groups_eeg = np.concatenate(groups_eeg)
            groups_eeg = groups_eeg.reshape(-1, groups_eeg.shape[-3], groups_eeg.shape[-2], groups_eeg.shape[-1])
            groups_eeg = torch.tensor(groups_eeg, dtype=torch.float, device=args.device)

            groups_gsr = np.concatenate(groups_gsr)
            groups_gsr = groups_gsr.reshape(-1, groups_gsr.shape[-3], groups_gsr.shape[-2], groups_gsr.shape[-1])
            groups_gsr = torch.tensor(groups_gsr, dtype=torch.float, device=args.device)

            out_eeg, out_eeg_encoder = model_eeg(groups_eeg, mode='contrast')
            out_gsr, out_gsr_encoder = model_gsr(groups_gsr, mode='contrast')
            if args.N > 1:
                out_eeg = torch.reshape(out_eeg, (-1, args.N, out_eeg.shape[-1]))
                out_eeg = torch.max(out_eeg, dim=1)[0]
                out_gsr = torch.reshape(out_gsr, (-1, args.N, out_gsr.shape[-1]))
                out_gsr = torch.max(out_gsr, dim=1)[0]

                out_eeg_encoder = torch.reshape(out_eeg_encoder, (-1, args.N, out_eeg_encoder.shape[-1]))
                out_eeg_encoder = torch.max(out_eeg_encoder, dim=1)[0]
                out_gsr_encoder = torch.reshape(out_gsr_encoder, (-1, args.N, out_gsr_encoder.shape[-1]))
                out_gsr_encoder = torch.max(out_gsr_encoder, dim=1)[0]
            c_con_loss = model_cc(out_eeg_encoder, out_gsr_encoder, criterion, args.tau)

            tem_loss_eeg, lab_con_eeg, logits_ab_eeg = constrast_loss(out_eeg, criterion, args.tau)
            _, log_p_eeg = torch.max(logits_ab_eeg.data, 1)
            tem_loss_gsr, lab_con_gsr, logits_ab_gsr = constrast_loss(out_gsr, criterion, args.tau)
            _, log_p_gsr = torch.max(logits_ab_gsr.data, 1)

            loss = (0.5 * tem_loss_eeg) + (0.5 * tem_loss_gsr) + c_con_loss

            evaluation_batch_eeg = ((log_p_eeg == lab_con_eeg).cpu().numpy() * 1)
            evaluation_batch_gsr = ((log_p_gsr == lab_con_gsr).cpu().numpy() * 1)
            acc_eeg = sum(evaluation_batch_eeg) / evaluation_batch_eeg.shape[0]
            acc_gsr = sum(evaluation_batch_gsr) / evaluation_batch_gsr.shape[0]
            total_acc_eeg += acc_eeg
            total_acc_gsr += acc_gsr
            total_loss_eeg += tem_loss_eeg
            total_loss_gsr += tem_loss_gsr
        test_acces_eeg = total_acc_eeg / iters
        test_acces_gsr = total_acc_gsr / iters
        test_losses_eeg = total_loss_eeg / iters
        test_losses_gsr = total_loss_gsr / iters
    return test_acces_eeg, test_losses_eeg, test_acces_gsr, test_losses_gsr
