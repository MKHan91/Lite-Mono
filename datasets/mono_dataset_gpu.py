from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import h5py as h5


# region - MonoDataset
class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders"""
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.Resampling.LANCZOS

        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            self.color_jitter = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
            
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.load_depth = self.check_depth()
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        # self.kitti_hdf5 = h5.File("/home/dev/Lite_Mono/datasets/kitti_data/kitti.hdf5", 'r')
        self.kitti_hdf5 = None

    # region - len
    def __len__(self):
        return len(self.filenames)


    # region - getitem
    def __getitem__(self, index):
        assert self.kitti_hdf5 is not None
        
        inputs = {}

        line = self.filenames[index].split()
        folder = line[0]
        folderName = folder.split('/')[-1]

        if len(line) == 3:
            frame_index = int(line[1])
            side = line[2]
        else:
            frame_index = 0
            side = None
            
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        if do_color_aug:
            color_aug = self.color_jitter
        else:
            color_aug = (lambda x: x)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        for frame_idx in self.frame_idxs:
            for scale in range(self.num_scales):
                dataset = self.kitti_hdf5['dataset']
                
                image = dataset[f'{folderName}_image_0{self.side_map[side]}_{frame_index}_{frame_idx}_{scale}'][:]
                # image = Image.fromarray(image)
                
                if do_flip:
                    image = np.fliplr(image).copy()
                    
                image_tensor = self.to_tensor(image)
                # image_tensor_aug = self.to_tensor(color_aug(image))
                image_tensor_aug = color_aug(image_tensor)
                
                inputs[('color', frame_idx, scale)] = image_tensor
                inputs[('color_aug', frame_idx, scale)] = image_tensor_aug

        if self.load_depth:
            depth_gt = dataset[f'{folderName}_velodyne_points_{self.side_map[side]}_{frame_index}'][:]
            if do_flip:
                depth_gt = np.fliplr(depth_gt)
                
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs


    def check_depth(self):
        raise NotImplementedError
    

