import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# PyTorch를 사용하여 컬러 이미지의 L 채널을 흑백으로 변환하고,
# 해당 이미지를 학습 및 검증을 위한 데이터로 처리하기 위한 데이터셋 및 데이터로더를 정의한다.

SIZE = 256 # 이미지의 크기를 256x256으로 설정한다.

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train': # 학습 데이터일 경우
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC), # 이미지의 크기를 256x256으로 설정한다.
                transforms.RandomHorizontalFlip(),  # 무작위로 이미지를 수평으로 뒤집어 데이터 증강을 수행한다.
            ])
        elif split == 'val': # 검증 데이터일 경우
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC) # 이미지의 크기를 256x256으로 설정한다.

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB") # 이미지를 RGB로 변환한다.
        img = self.transforms(img)
        img = np.array(img) # PIL 이미지를 Numpy 배열로 변환한다.
        img_lab = rgb2lab(img).astype("float32")  # RGB 이미지를 L*a*b로 변환한다.
        img_lab = transforms.ToTensor()(img_lab) # LAB 이미지를 PyTorch Tensor로 변환한다.
        L = img_lab[[0], ...] / 50. - 1.  # L 채널을 -1에서 1로 정규화한다.
        ab = img_lab[[1, 2], ...] / 110.  # A, B 채널을 -1에서 1로 정규화한다.

        return {'L': L, 'ab': ab} # 딕셔너리 형태로 변환

    def __len__(self):
        return len(self.paths)