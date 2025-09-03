import io


import lmdb
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class LMDBDataset(Dataset):
    def __init__(self, gt_path, image_path, transform=None):
        self.env_gt = lmdb.open(gt_path,
                                subdir=True,         # LMDB가 디렉터리면 True
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.env_image = lmdb.open(image_path,
                                subdir=True,         # LMDB가 디렉터리면 True
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        
        self.image_keys = []
        self.gt_keys = []
        with self.env_image.begin(write=False) as image_txn:
            with image_txn.cursor() as cursor:
                for key, value in cursor:
                    img_size_list, _ = pickle.loads(value)
                    self.image_keys.append(key)
                    
        with self.env_gt.begin(write=False) as gt_txn:
            self.length = pickle.loads(gt_txn.get(b'__len__'))
            with gt_txn.cursor() as cursor:
                for key, _ in cursor:
                    self.gt_keys.append(key)
        
        
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_key = self.image_keys[idx:3]
        gt_key = self.gt_keys[idx // 3]
        
        with self.env_image.begin(write=False) as image_txn:
            value = image_txn.get(image_key)
            img = Image.open(io.BytesIO(value)).convert("RGB")

        with self.env_gt.begin(write=False) as gt_txn:
            value = gt_txn.get(gt_key)
            gt = np.load(io.BytesIO(value))
            gt = torch.from_numpy(gt).unsqueeze(0)
            
        if self.transform:
            img = self.transform(img)

        return img, gt

# 사용 예시
if __name__ == "__main__":
    dataset = LMDBDataset(gt_path="/home/dev/Lite_Mono/datasets/kitti_data/kitti_gt.lmdb",
                          image_path="/home/dev/Lite_Mono/datasets/kitti_data/kitti_image.lmdb",
                          transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    for images, labels in loader:
        print(images.shape, labels)
