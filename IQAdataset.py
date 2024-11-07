import torch
from torch.utils.data import Dataset
import glob
import os.path as osp
import torchvision.transforms.functional as F
import random

class MyDataset(Dataset):
    def __init__(self, data, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode
        self.dataset = glob.glob(osp.join('pt_bid_org/*.pt'))
        self.img_reads = []
        self.label_array = []
        split = data
        
        print(f"Dataset length: {len(self.dataset)}")

        for i in split:
            f_l = torch.load(self.dataset[i])
            fig = f_l['rgb']
            lab = f_l['label']
            self.img_reads.append(fig)
            self.label_array.append(lab)

        assert len(self.img_reads) == len(self.label_array)
        print(f'{mode.capitalize()} 数据集大小: {len(self.img_reads)}')

    def __getitem__(self, index):
        if self.mode == 'train':
            # 计算当前样本对应的图像索引和patch索引
            img_index = index // 3  # 每个图像对应3个patch
            img = self.img_reads[img_index]
            label = self.label_array[img_index]

            # 随机裁剪一个224x224的patch
            img_patch = F.crop(img, 
                               top=random.randint(0, img.size(1) - 224), 
                               left=random.randint(0, img.size(2) - 224), 
                               height=224, 
                               width=224)
            if torch.rand(1).item() > 0.5:  # 随机水平翻转
                img_patch = F.hflip(img_patch)

            return img_patch, label

        else:  # 测试模式
            img_index = index  # 每个图像对应5个patch
            img = self.img_reads[img_index]
            label = self.label_array[img_index]

            # 随机裁剪5个224x224的patch
            patches = []
            for _ in range(5):
                img_patch = F.crop(img, 
                                   top=random.randint(0, img.size(1) - 224), 
                                   left=random.randint(0, img.size(2) - 224), 
                                   height=224, 
                                   width=224)
                patches.append(img_patch) 

            return torch.stack(patches), label

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_reads) * 3  # 训练模式：总样本数是原始图像数的三倍
        else:
            return len(self.img_reads)  # 测试模式：总样本数是原始图像数的五倍

def train_dataset(split_tra):
    return MyDataset(data=split_tra, mode='train')

def val_dataset(split_val):
    return MyDataset(data=split_val, mode='test')
