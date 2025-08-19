import torch
import numpy as np
from tqdm import tqdm
import shutil
import torch.nn.functional as F
import copy
import random

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # eq_out = (torch.abs(output - output[:, 0].repeat(output.shape[1], 1).t()) < 1e-5).sum(axis=1)
        # print(eq_out)
        # eq_num = (eq_out > 1).sum()
        # if eq_out.sum() > len(eq_out):
        #     print('Equal logits for different entries!')
        #     print(output[eq_out>1, :].shape, output[eq_out>1, :])

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # if k == 1:
            #     correct_k = correct_k - eq_num
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def info_nce_loss(args, features):
    bs = int(features.shape[0] // 2)
    labels = torch.cat([torch.arange(bs) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    # Normlize the features according to subject
    if args.stratified == 'stratified':
        features_str = features.clone()
        features_str[:bs, :] = (features[:bs, :] - features[:bs, :].mean(
            dim=0)) / (features[:bs, :].std(dim=0) + 1e-3)
        features_str[bs:, :] = (features[bs:, :] - features[bs:, :].mean(
            dim=0)) / (features[bs:, :].std(dim=0) + 1e-3)
        features = F.normalize(features_str, dim=1)
    elif args.stratified == 'bn':
        features_str = features.clone()
        features_str = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-3)
        features = F.normalize(features_str, dim=1)
    elif args.stratified == 'no':
        features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    one_indies = torch.where(labels == 1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # print('positives:', positives.shape)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # logits = torch.cat([positives, negatives], dim=1)
    # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    # Put the positive column at the end (when all the entries are the same, the top1 acc will be 0; while if the
    # positive column is at the start, the top1 acc might be exaggerated)
    logits = torch.cat([negatives, positives], dim=1)
    # The label means the last column contain the positive pairs
    labels = torch.ones(logits.shape[0], dtype=torch.long) * (logits.shape[1] - 1)
    labels = labels.to(args.device)

    logits = logits / args.temperature
    return logits, labels
def data_aug(data, label):
    data_a = []
    label_a = []

    for i in tqdm(range(data.shape[0]), desc="Data Augmentataon"):
        data_in = []
        label_in = []
        for j in range(data.shape[1]):
            data_sub = []
            label_sub = []
            data_sample = data[i, j, :, :]
            label_sample = np.array(label[i, j])
            data_sub.append(data_sample)
            data_sub.append(scaled_sample(data_sample, 0.7, 0.8))
            data_sub.append(scaled_sample(data_sample, 1.2, 1.3))
            data_sub.append(add_gauss_noise(data_sample, snr=5))
            data_sub.append(add_gauss_noise(scaled_sample(data_sample, 0.7, 0.8), snr=5))
            data_sub.append(add_gauss_noise(scaled_sample(data_sample, 1.2, 1.3), snr=5))
            for _ in range(6):
                label_sub.append(label_sample)
            data_sub = np.stack(data_sub, axis=0)
            label_sub = np.stack(label_sub)
            data_in.append(data_sub)
            label_in.append(label_sub)
        data_in = np.stack(data_in, axis=0).transpose(1, 0, 2, 3, 4)
        label_in = np.stack(label_in).transpose(1,0)
        data_a.append(data_in)
        label_a.append(label_in)
    data_a = np.stack(data_a, axis=0)
    data_a = data_a.reshape(-1, *data_a.shape[2:])
    label_a = np.stack(label_a, axis=0)
    label_a = label_a.reshape(-1, label_a.shape[-1])
    return data_a, label_a


def scaled_sample(sample, min, max):
    sample = copy.deepcopy(sample)  # need to not modify directly passed sample
    # Compute scaling factor
    alpha = random.uniform(min, max)
    # Apply same scaling factor to all modalities
    # Physio modalities: apply same scaling factor to all channels
    sample = alpha * sample
    return sample

def add_gauss_noise(sample, snr):
    sample = copy.deepcopy(sample)  # need to not modify directly passed sample
    sample_size = sample.shape  #
    # Create tensor with physio size, 1 ch, but initialized with standard normal distribution
    gaussian_noise = torch.empty(sample_size).normal_(mean=0, std=1)
    # Apply same noise to all modalities, but with std needed to obtain given SNR -------------------
    # Physio modalities: apply different std for modality

    noise = gaussian_noise.numpy()
    sample = sample + std_for_SNR(sample, noise, snr) * noise
    return sample

def std_for_SNR(signal, noise, snr):
    signal = signal.reshape(signal.shape[0]*signal.shape[1], -1)
    noise = noise.reshape(noise.shape[0] * noise.shape[1], -1)
    '''Compute the gain to be applied to the noise to achieve the given SNR in dB'''
    signal_power = np.var(signal)
    noise_power = np.var(noise)
    # Gain to apply to noise, from SNR formula: SNR=10*log_10(signal_power/noise_power)
    g = np.sqrt(10.0 ** (-snr/10) * signal_power / noise_power)
    return g
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

