import random
import torch
import numpy as np
from torchvision import transforms
# from kornia.augmentation import kornia



class AugmentationOps:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # self.color_aug = torch.nn.Sequential(
        #     kornia.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
        # ).to(self.device)
        
        
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def adjust_brightness(self, image, factor=(0.8, 1.2)):
        brightness_factor = random.uniform(factor)
        image = (image * brightness_factor).clamp(0., 255.)
        
        return image
        
        
    
    def forward(self, inputs):
        scale0_img = inputs[('color', -1, 0)].to(self.device).type(torch.float32), inputs[('color', 0, 0)].to(self.device).type(torch.float32), inputs[('color', 1, 0)].to(self.device).type(torch.float32)
        scale1_img = inputs[('color', -1, 1)].to(self.device).type(torch.float32), inputs[('color', 0, 1)].to(self.device).type(torch.float32), inputs[('color', 1, 1)].to(self.device).type(torch.float32)
        scale2_img = inputs[('color', -1, 2)].to(self.device).type(torch.float32), inputs[('color', 0, 2)].to(self.device).type(torch.float32), inputs[('color', 1, 2)].to(self.device).type(torch.float32)
        scale3_img = inputs[('color', -1, 3)].to(self.device).type(torch.float32), inputs[('color', 0, 3)].to(self.device).type(torch.float32), inputs[('color', 1, 3)].to(self.device).type(torch.float32)
        
        # (num, batch, height, width, channel)
        scale0_img = torch.stack(scale0_img, dim=0)
        scale1_img = torch.stack(scale1_img, dim=0)
        scale2_img = torch.stack(scale2_img, dim=0)
        scale3_img = torch.stack(scale3_img, dim=0)
        
        # region Hflip
        # ------------------------- horziontal flip -------------------------
        do_flip = random.random() > 0.5
        if do_flip:
            scale0_img = torch.flip(scale0_img, dims=[3])
            scale1_img = torch.flip(scale1_img, dims=[3])
            scale2_img = torch.flip(scale2_img, dims=[3])
            scale3_img = torch.flip(scale3_img, dims=[3])
        # -------------------------------------------------------------------

        # region ColorJitter
        # ------------------------- Color Jitter ----------------------------
        # brightness_factor = (0.8, 1.2)
        # contrast = (0.8, 1.2)
        # saturation = (0.8, 1.2)
        # hue = (-0.1, 0.1)
        scale0_img = self.adjust_brightness(scale0_img, factor=(0.8, 1.2))
        scale1_img = self.adjust_brightness(scale1_img, factor=(0.8, 1.2))
        scale2_img = self.adjust_brightness(scale2_img, factor=(0.8, 1.2))
        scale3_img = self.adjust_brightness(scale3_img, factor=(0.8, 1.2))
        
        mean = scale0_img.mean(dim=(2,3), keepdim=True)
        # -------------------------------------------------------------------
        
        
        return 