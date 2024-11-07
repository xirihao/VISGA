import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scipy.io
from scipy import stats
import numpy as np
from progressbar import *
import torch
import torch.nn.functional as F
from shutil import copyfile
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.optim as optim
from pathlib import Path
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus'] = False
from model import netReg
import torch.nn as nn
from IQAdataset import train_dataset, val_dataset
import time
import random


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def char_shijian(t):
    if t >= 3600:
        hour_t = int(t / 3600)
    else:
        hour_t = 0
    t = t - hour_t * 3600
    if t >= 60:
        minite_t = int(t / 60)
    else:
        minite_t = 0
    t = t - minite_t * 60
    return ('%d小时%d分钟%.4f秒' % (hour_t, minite_t, t))

def rmse_function(a, b):
    """does not need to be array-like"""
    a = np.array(a)
    b = np.array(b)
    mse = ((a-b)**2).mean()
    rmse = math.sqrt(mse)
    return rmse

def get_perform(perf_dict, save_folder):

    val_srcc_mean = perf_dict['val_srcc_mean']
    val_plcc_mean = perf_dict['val_plcc_mean']
    val_krcc_mean = perf_dict['val_krcc_mean']
    val_rmse_mean = perf_dict['val_rmse_mean']

    val_srcc_mean = np.array(val_srcc_mean)
    theIndex = np.argmax(val_srcc_mean)

    val_srcc_mean_max = val_srcc_mean[theIndex]
    val_plcc_mean_max = val_plcc_mean[theIndex]
    val_krcc_mean_max = val_krcc_mean[theIndex]
    val_rmse_mean_max = val_rmse_mean[theIndex]

    return val_srcc_mean_max, val_plcc_mean_max, val_krcc_mean_max, val_rmse_mean_max

class Trainer(object):
    def __init__(self, data_loader_train, data_loader_val, regmodel, save_folder, LR, plot_epoch, train_patience):

        self.train_loader = data_loader_train
        self.val_loader = data_loader_val

        self.epochs = 30
        self.start_epoch = 0
        self.use_gpu = True
        self.counter = 0
        self.train_patience = train_patience

        self.plot_epoch = plot_epoch
        self.model = regmodel
        print('模型包含%d个参数' % sum([p.data.nelement() for p in self.model.parameters()]))
        
        self.adam_lr = LR
        print('使用的学习率是%.1e' % self.adam_lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_lr)

        # 指定文件夹, 和哪一个模型状态(最新/最佳)
        self.resume = False
        if self.resume == False:
            self.saveFolder = save_folder
            self.lr_bin = []
            self.counter_bin = []

            self.train_loss_bin = []
            self.train_srcc_bin = []
            self.train_plcc_bin = []

            self.val_loss_bin = []
            self.val_srcc_bin = []
            self.val_plcc_bin = []

            self.val_srcc_mean_bin = []
            self.val_plcc_mean_bin = []
            self.val_krcc_mean_bin = []
            self.val_rmse_mean_bin = []

            self.is_best = True
            self.is_best_val_srcc_mean = True
            self.is_best_loss = True
        else:
            self.resumeFolder = ''
            self.saveFolder = self.resumeFolder

            model_path = 'e'
            self.ckpt_path = os.path.join(self.resumeFolder, model_path)
            print('恢复上次训练, 从%s中提取模型' % self.ckpt_path)

            # 加载模型
            ckpt = torch.load(self.ckpt_path)
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optim_state'])

            # 加载性能字典
            dictPath = os.path.join(self.resumeFolder, 'perform_dict.txt')
            lrPath = os.path.join(self.resumeFolder, 'lr_dict.txt')

            with open(dictPath, 'r') as f:
                perf = f.read()
            perf_dict = eval(perf)

            with open(lrPath, 'r') as f:
                lr = f.read()
            lr_list = eval(lr)

            self.lr_bin = lr_list

            self.train_loss_bin = perf_dict['train_loss']
            self.train_srcc_bin = perf_dict['train_srcc']
            self.train_plcc_bin = perf_dict['train_plcc']

            self.val_loss_bin = perf_dict['val_loss']
            self.val_srcc_bin = perf_dict['val_srcc']
            self.val_plcc_bin = perf_dict['val_plcc']


            self.is_best = False
            self.is_best_val_srcc_mean = False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))

            train_loss, train_srcc, train_plcc = self.train_one_epoch()
            val_loss, val_srcc_mean, val_plcc_mean, val_krcc_mean, val_rmse_mean = self.validate(epoch)

            temp_lr = self.optimizer.param_groups[0]['lr']
            print('当前学习率%.1e' % temp_lr)
            self.lr_bin.append(temp_lr)
            self.counter_bin.append(self.counter)

            f = open('%s/lr_dict.txt' % self.saveFolder, 'w')
            f.write(str(self.lr_bin))
            f.close()

            # 打印当epoch的性能
            print('train loss: %.3f, val loss: %.3f' % (train_loss, val_loss))
            msg1 = 'Train>>>  srcc: {0:6.4f}  plcc: {1:6.4f}   VAL>>> srcc: {2:6.4f}  plcc: {3:6.4f}  krcc: {4:6.4f}  rmse: {5:6.4f}'
            msg2 = '                                  VAL_MEAN>>> srcc: {0:6.4f}  plcc: {1:6.4f}   BEST_VAL_MEAN>>> srcc: {2:6.4f}  plcc: {3:6.4f}'

            if epoch >= self.start_epoch + 1:
                self.is_best_loss = (train_loss < np.array(self.train_loss_bin).min())
                self.is_best_val_srcc_mean = (val_srcc_mean > np.array(self.val_srcc_mean_bin).max())

            if self.is_best_loss:
                self.counter = 0
                self.best_val_srcc_mean = val_srcc_mean
                msg2 += '[^]'
            else:
                self.counter += 1
                print('已经%d个epoch loss没有下降' % self.counter)

            self.save_checkpoint({'epoch': epoch,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict(),
                                  'best_val_srcc_mean': self.best_val_srcc_mean},
                                 self.is_best_val_srcc_mean)

            # 记录相关数据
            self.train_loss_bin.append(train_loss)
            self.train_srcc_bin.append(train_srcc)
            self.train_plcc_bin.append(train_plcc)

            self.val_loss_bin.append(val_loss)
            self.val_srcc_mean_bin.append(val_srcc_mean)
            self.val_plcc_mean_bin.append(val_plcc_mean)
            self.val_krcc_mean_bin.append(val_krcc_mean)
            self.val_rmse_mean_bin.append(val_rmse_mean)

            print(msg1.format(train_srcc, train_plcc, val_srcc_mean, val_plcc_mean, val_krcc_mean, val_rmse_mean))
            print(msg2.format(val_srcc_mean, val_plcc_mean, self.val_srcc_mean_bin[self.val_plcc_mean_bin.index(np.array(self.val_plcc_mean_bin).max())], np.array(self.val_plcc_mean_bin).max()))

            perform_dict = {'train_loss': self.train_loss_bin, 'train_srcc': self.train_srcc_bin, 'train_plcc': self.train_plcc_bin,
                            'val_loss': self.val_loss_bin,
                            'val_srcc_mean': self.val_srcc_mean_bin, 'val_plcc_mean': self.val_plcc_mean_bin, 'val_krcc_mean': self.val_krcc_mean_bin, 'val_rmse_mean': self.val_rmse_mean_bin,
                            'lr': self.lr_bin, 'counter': self.counter_bin}

            f = open('%s/perform_dict.txt' % self.saveFolder, 'w')
            f.write(str(perform_dict))
            f.close()

            if (epoch+1) % 30 == 0:
                val_srcc_mean_max, val_plcc_mean_max, val_krcc_mean_max, val_rmse_mean_min = get_perform(perform_dict, self.saveFolder)

                print('>>>>>   SRCC: %.4f,  PLCC: %.4f,  KRCC: %.4f,  RMSE: %.4f   <<<<<' % (
                        val_srcc_mean_max, val_plcc_mean_max, val_krcc_mean_max, val_rmse_mean_min))
                print('[!]已经%d个epoch没有性能提升, 停止训练' % self.train_patience)
                return val_plcc_mean_max

    def train_one_epoch(self):
        self.model.train()
        loss_bin = AverageMeter()
        batch_time = AverageMeter()
        predict_bin = []
        y_bin = []

        tic = time.time()
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Counter(), ' ', ETA(), ' ', FileTransferSpeed()]
        progress = ProgressBar(widgets=widgets)
        for i, (imgRead, y) in enumerate(progress(self.train_loader)):
            if self.use_gpu:
                imgRead= imgRead.cuda()
                y = y.cuda()
            y = y.float()

            self.batch_size = imgRead.shape[0]

            # 前向传播
            predict = self.model(imgRead)
            predict = predict.squeeze()

            # 损失函数, 以及反向传播, 更新梯度
            loss = F.mse_loss(predict, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_bin.update(loss.cpu().item())

            if predict.dim() == 0:  # 如果是标量
                predict_bin.append(predict.item())  # 将标量转换为 Python 标量并添加到列表
            else:
                predict_bin.extend(predict.detach().cpu().numpy())
            y_bin.extend(y.cpu().numpy())

            # 计算用的时间
            toc = time.time()
            batch_time.update(toc-tic)

        srcc = stats.spearmanr(predict_bin, y_bin)[0]
        plcc = stats.pearsonr(predict_bin, y_bin)[0]

        return loss_bin.avg, srcc, plcc

    def validate(self, epoch):
        self.model.eval()
        loss_bin = AverageMeter()
        batch_time = AverageMeter()
        predict_bin = []
        y_bin = []

        tic = time.time()
        with torch.no_grad():
            widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Counter(), ' ', ETA(), ' ', FileTransferSpeed()]
            progress = ProgressBar(widgets=widgets)
            for i, (imgRead, y) in enumerate(progress(self.val_loader)):
                if self.use_gpu:
                    imgRead= imgRead.cuda()
                    y = y.cuda()
                y = y.float()

                self.batch_size = imgRead.shape[0]

                predict = self.model(imgRead.squeeze())
                predict = predict.mean()

                # 损失函数, 不再反向传播和更新梯度
                loss = F.mse_loss(predict, y)
                loss_bin.update(loss.item())

                # 收集数据
                if predict.dim() == 0:  # 如果是标量
                    predict_bin.append(predict.item())  # 将标量转换为 Python 标量并添加到列表
                else:
                    predict_bin.extend(predict.detach().cpu().numpy())
                y_bin.extend(y.cpu().numpy())

                # 计算时间
                toc = time.time()
                batch_time.update(toc-tic)


        predict_bin = np.array(predict_bin)
        y_bin = np.array(y_bin)
        srcc_mean = stats.spearmanr(predict_bin, y_bin)[0]
        plcc_mean = stats.pearsonr(predict_bin, y_bin)[0]
        krcc_mean = stats.stats.kendalltau(predict_bin, y_bin)[0]
        rmse_mean = rmse_function(predict_bin, y_bin)

        return loss_bin.avg, srcc_mean, plcc_mean, krcc_mean, rmse_mean

    def save_checkpoint(self, state, is_best_val_srcc_mean):
        if is_best_val_srcc_mean:
            filename = 'model_BEST_SRCC.pth.tar'
            ckpt_path = os.path.join(self.saveFolder, filename)
            torch.save(state, ckpt_path)

    def save_checkpoint_last(self, epoch, state):
        filename = 'model_last_ep%d.pth.tar' % epoch
        ckpt_path = os.path.join(self.saveFolder, filename)
        torch.save(state, ckpt_path)


def main():  # 单独使用main函数是为了避免使用全局变量, 是接口更严谨
    regmodel = netReg()

    regmodel.cuda()
    ModelName = regmodel.model_name

    JiaZai = False
    batchSize = 32
    LR = 5e-5
    TRAIN_PATIENCE = 3
    train_file_name = Path(__file__).name

    bigFolder = time.strftime('bs%d_pt_xiaolr%d' % (batchSize, TRAIN_PATIENCE))
    if not os.path.exists(bigFolder):
        os.mkdir(bigFolder)
    saveFolder = bigFolder + '/' + train_file_name[:-3] + ModelName + \
                 '_LR%.1e_T%s' % (LR, time.strftime('%m%d_%H%M%S'))
    print('训练，模型，数据的代码已备份。')

    split_all = torch.randperm(586)
    print(split_all)
    split_tra = split_all[:int(0.8 * 586)]
    split_val = split_all[int(0.8 * 586):]
    trainDataset = train_dataset(split_tra)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, pin_memory=True)

    valDataset = val_dataset(split_val)
    valLoader = DataLoader(valDataset, batch_size=1, shuffle=False, pin_memory=False)
    PLOT_EPOCH = 5
    
    trainer = Trainer(trainLoader, valLoader, regmodel, saveFolder, LR, PLOT_EPOCH, TRAIN_PATIENCE)
    trainer.train()

if __name__ == '__main__':
    global_tic = time.time()
    main()
    global_toc = time.time()
    print('用时%s' % char_shijian(global_toc-global_tic))


