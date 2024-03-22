import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
from tqdm import tqdm
import math
import os
import pandas as pd
from dataset import SUGCsal
import shutil
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import VQA_Network
import time


def train(epoch, model, optimizer, scheduler, train_loader, device, writer):
    model.train()
    loss_sum = 0
    for i, data in enumerate(tqdm(train_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        data['frame_feature'] = data['frame_feature'].to(device)
        data['sal_feature'] = data['sal_feature'].to(device)
        y = data["label"].float().detach().to(device).unsqueeze(-1)
        y_pred = model(data)
        p_loss = plcc_loss(y_pred, y)
        r_loss = rank_loss(y_pred, y)
        loss = p_loss + 0.3 * r_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar("train/loss", loss, epoch * len(train_loader) + i)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar("train/lr", lr, epoch * len(train_loader) + i)

        loss_sum += loss

    print('Epoch {}::::loss: {:.3f}'.format(epoch, loss_sum/len(train_loader)))


def valid(epoch, model, valid_loader, device, writer, best, training_runs_dir, test_loader):
    model.eval()

    best_s, best_p, best_k, best_r = best
    predictions = []
    labels = []
    videos =[]

    for i, data in enumerate(tqdm(valid_loader, desc="Validating")):
        avg_result = []
        with torch.no_grad():

            for zz in range(len(data['frame_feature'])):
                new_data = {}
                new_data['feat'] = data['feat'].to(device)
                new_data['frame_feature'] = data['frame_feature'][zz].to(device)
                new_data['sal_feature'] = data['sal_feature'][zz].to(device)
                result = model(new_data).cpu().numpy()
                avg_result.append(result[0][0])

        labels.append(data["label"].item())
        videos.append(data["name"][0])
        predictions.append(np.mean(avg_result).item())

    s = spearmanr(labels, predictions)[0]
    p = pearsonr(labels, predictions)[0]
    k = kendallr(labels, predictions)[0]
    zz = [(x-y) ** 2 for x, y in zip(labels, predictions)]
    r = np.sqrt(np.mean(zz))

    predictions = rescale(predictions, gt=labels)
    data = {'filename': videos, 'score': predictions}  #
    df = pd.DataFrame(data)

    df.to_csv(training_runs_dir + '/validation_{:2d}_{:.3f}_{:.3f}.csv'.format(epoch, p, s), index=False)


    writer.add_scalar("valid/plcc", p, epoch)
    writer.add_scalar("valid/srcc", s, epoch)
    writer.add_scalar("valid/krcc", k, epoch)

    print('Epoch {}, SRCC:{:.3f} PLCC:{:.3f} KRCC:{:.3f} RMSE:{:.3f}'.format(epoch, s, p, k, r))
    if s + p > best_s + best_p :
        state_dict = model.state_dict()
        torch.save(
                {"state_dict": state_dict, "validation_results": best,},
                f"{training_runs_dir}/SUGC_{epoch}.pth")

        best_s, best_p, best_k, best_r = (s, p, k, r)
        print("best_SRCC {:.3f}, best_PLCC {:.3f}, best_KRCC {:.3f}, vbest_RMSE {:.3f}".format(best_s, best_p, best_k, best_r))
        time.sleep(1)
        if epoch > 2:
            test(model, device, training_runs_dir, test_loader, epoch, s, p, k)

    return best_s, best_p, best_k, best_r


def test(model, device, training_runs_dir, test_loader, epoch, s, p, k):
    model.eval()
    predictions = []
    labels = []
    videos = []

    for i, data in enumerate(tqdm(test_loader, desc="Testing")):
        avg_result = []
        with torch.no_grad():

            for zz in range(len(data['frame_feature'])):
                new_data = {}
                new_data['feat'] = data['feat'].to(device)
                new_data['frame_feature'] = data['frame_feature'][zz].to(device)
                new_data['sal_feature'] = data['sal_feature'][zz].to(device)
                result = model(new_data).cpu().numpy()
                avg_result.append(result[0][0])

        labels.append(data["label"].item())
        videos.append(data["name"][0])
        predictions.append(np.mean(avg_result).item())

    data = {'filename': videos, 'score': predictions}  #
    df = pd.DataFrame(data)

    df.to_csv(training_runs_dir + '/prediction_{:2d}_{:.3f}_{:.3f}.csv'.format(epoch, p, s), index=False)


def main(config):

    training_runs_dir = 'runs/S-UGC/' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    writer = SummaryWriter(training_runs_dir)

    copy_files = ['train.py', 'model.py', 'utils.py', 'dataset.py', 'swin_transformer_V2.py',
                   'convnext.py']
    os.makedirs(training_runs_dir + '/code/')
    for file in copy_files:
        shutil.copy2(file, training_runs_dir + '/code/')

    device = torch.device("cuda:" + config.gpu_id.split(',')[0])

    model = VQA_Network().to(device)
    # pretrained_dict = torch.load('/mnt/hdd1/wsj/code/SUGC-CLIP-multi/runs/MVQA/2024-03-18-00:38:27/SUGC_8.pth')
    # model.load_state_dict(pretrained_dict, strict=False)
    model = torch.nn.DataParallel(model)

    train_settings = {'phase': 'train', 'anno_file': config.train_anno_file,
                      'data_prefix': config.train_video_dir,
                      'data_prefix_3D': config.train_feature_dir,
                      'feature_type': 'SlowFast',
                      'clip_len': 5,
                      'clip_type': 'uniform',
                      }
    train_dataset = SUGCsal(train_settings)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)

    valid_settings = {'phase': 'valid', 'anno_file': config.valid_anno_file,
                      'data_prefix': config.valid_video_dir,
                      'data_prefix_3D': config.valid_feature_dir,
                      'feature_type': 'SlowFast',
                      'clip_len': 5,
                      'clip_type': 'uniform'}
    valid_dataset = SUGCsal(valid_settings)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers*2)

    test_settings = {'phase': 'test', 'anno_file': config.test_anno_file,
                     'data_prefix': config.test_video_dir,
                     'data_prefix_3D': config.test_feature_dir,
                     'feature_type': 'SlowFast',
                     'clip_len': 5,
                     'clip_type': 'uniform'}
    test_dataset = SUGCsal(test_settings)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers*2)

    param_groups = []
    for key, value in dict(model.named_children()).items():
        if "backbone" in key:
            param_groups += [
                {
                    "params": value.parameters(),
                    "lr": config.base_lr * config.lr_mult,
                }
            ]
        else:
            param_groups += [
                {"params": value.parameters(), "lr": config.base_lr}
            ]

    optimizer = torch.optim.AdamW(lr=config.base_lr, params=param_groups, weight_decay=config.weight_decay)

    warmup_iter = 0
    warmup_iter += int(config.warmup_epochs * len(train_loader))
    max_iter = int((config.num_epochs + config.l_num_epochs) * len(train_loader))
    lr_lambda = (
        lambda cur_iter: cur_iter / warmup_iter
        if cur_iter <= warmup_iter
        else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, )

    best_results = -1, -1, -1, 1999
    for epoch in range(config.num_epochs):
        train(epoch, model, optimizer, scheduler, train_loader, device, writer)
        scheduler.step()
        best_s, best_p, best_k, best_r = valid(epoch, model, valid_loader, device, writer, best_results, training_runs_dir, test_loader)
        best_results = best_s, best_p, best_k, best_r


if __name__ == '__main__':
    import os
    import warnings
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument('--base_lr', type=float, default=5e-5)  # 3e-5
    parser.add_argument('--lr_mult', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=float, default=2.5)
    parser.add_argument('--l_num_epochs', type=float, default=0)
    parser.add_argument('--gpu_id', type=str, default='0,1')
    parser.add_argument('--train_anno_file', type=str, default='/mnt/hdd1/wsj/SUGC_VQA/S-UGC/train_data.csv')
    parser.add_argument('--train_video_dir', type=str, default='/mnt/hdd1/wsj/SUGC_VQA/S-UGC/Train/')
    parser.add_argument('--train_feature_dir', type=str, default='/mnt/hdd1/wsj/code/KVQ-challenge/feature/SUGC/train/')
    parser.add_argument('--valid_anno_file', type=str, default='/mnt/hdd1/wsj/SUGC_VQA/S-UGC/train_data.csv')
    parser.add_argument('--valid_video_dir', type=str, default='/mnt/hdd1/wsj/SUGC_VQA/S-UGC/Train/')
    parser.add_argument('--valid_feature_dir', type=str, default='/mnt/hdd1/wsj/code/KVQ-challenge/feature/SUGC/train/')
    parser.add_argument('--test_anno_file', type=str, default='/mnt/hdd1/wsj/code/S-UGC-New/test.csv')
    parser.add_argument('--test_video_dir', type=str, default='/mnt/hdd1/wsj/SUGC_VQA/S-UGC/Test/')
    parser.add_argument('--test_feature_dir', type=str, default='/mnt/hdd1/wsj/code/KVQ-challenge/feature/SUGC/test/')


    args = parser.parse_args()
    main(args)

