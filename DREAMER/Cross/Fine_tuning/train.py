from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import matrix_percent
def main(model_ecg, model_eeg, model_classifier,
         optimizer, train_loader, test_loader, args):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 3,
                                                                     eta_min=0,
                                                                     last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss().to(args.device)
    train_losses = []
    train_acces = []
    test_loss, test_acc, cm, test_f1 = eval(model_ecg, model_eeg,
                                            model_classifier, test_loader, loss_fn)
    print(f'Epoch {0}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}, Test f1 {test_f1:.4f}')
    print(f'confusion_matrix:{cm}')

    for epoch in range(1, args.epochs+1):
        model_classifier.train()
        train_losses_tem, train_acces_tem = train(model_ecg, model_eeg,
                                                  model_classifier, train_loader, optimizer, loss_fn)
        train_acces.append(train_acces_tem)
        train_losses.append(train_losses_tem)
        print(f'Epoch {epoch}, Train loss {train_losses_tem:.4f}, Train acc {train_acces_tem:.4f}')
        test_loss, test_acc, cm, test_f1 = eval(model_ecg, model_eeg,
                                                model_classifier, test_loader, loss_fn)
        print(f'Epoch {epoch}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}, Test f1 {test_f1:.4f}')
        print(f'confusion_matrix:{cm}')

        scheduler.step()

        print('learning rate', scheduler.get_lr()[0])
    test_loss, test_acc, cm, test_f1 = eval(model_ecg, model_eeg,
                                            model_classifier, test_loader, loss_fn)
    print(f'Test loss {test_loss:.4f}, Test acc {test_acc:.4f}, Test f1 {test_f1:.4f}')
    print(f'confusion_matrix:{cm}')

    return test_acc, test_f1
def eval(model_ecg, model_eeg, model_classifier, test_loader, loss_fn):
    model_ecg.eval()
    model_eeg.eval()
    model_classifier.eval()
    total_cm, total_loss, total_acc, total_f1, iter = 0, 0, 0, 0, 0
    with torch.no_grad():
        for pair in test_loader:
            x_eeg, x_ecg, y = pair[0], pair[1], pair[2]
            x_eeg = x_eeg.cuda().float().contiguous()
            x_ecg = x_ecg.cuda().float().contiguous()
            y = y.cuda().long().contiguous()

            out_eeg = model_eeg(x_eeg, mode='embedding')
            out_ecg = model_ecg(x_ecg, mode='embedding')

            out = model_classifier(out_eeg, out_ecg)
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
            # f1_sco = f1_score(y_true, y_pred, average='weighted') # dimension == "four"
            f1_sco = f1_score(y_true, y_pred)
            total_f1 += f1_sco
            total_cm += cm
    test_loss = total_loss / iter
    test_acc = total_acc / iter
    f_cm = total_cm/iter
    test_f1 = total_f1/iter
    return test_loss, test_acc, f_cm, test_f1

#Training function
def train(model_ecg, model_eeg, model_classifier, train_loader, optimizer, loss_fn):
    model_ecg.train()
    model_eeg.train()
    model_classifier.train()
    train_losses_tem = []
    train_acces_tem = []
    i = 0
    for pair in tqdm(train_loader, desc='Processing', unit='pairs'):
        x_eeg, x_ecg, y = pair[0], pair[1], pair[2]
        x_eeg = x_eeg.cuda().float().contiguous()
        x_ecg = x_ecg.cuda().float().contiguous()
        y = y.cuda().long().contiguous()

        out_eeg = model_eeg(x_eeg, mode='embedding')
        out_ecg = model_ecg(x_ecg, mode='embedding')

        out = model_classifier(out_eeg, out_ecg)
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
