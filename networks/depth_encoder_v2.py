import math
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.cuda

from networks import core_layer as core
from networks import custom_layers as clayers



# region - Main Arch
class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):
        super().__init__()

        # if model == 'lite-mono':
        self.num_ch_enc = np.array([48, 80, 128])
        self.depth = [4, 4, 10]
        self.dims = [48, 80, 128]
        self.asym_dims = [64, 128, 256]

        if height == 192 and width == 640:
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            # self.dilation = [[1, 1, 2], [1, 1, 2], [1, 1, 2, 1, 1, 2, 1, 2, 3]]
        elif height == 320 and width == 1024:
            self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']


        self.avg_pool2 = clayers.AvgPool(ratio=2)
        self.avg_pool4 = clayers.AvgPool(ratio=4)
        self.avg_pool8 = clayers.AvgPool(ratio=8)

        
        self.init_conv = nn.Sequential(
            clayers.StandardConv(in_chans, self.dims[0],
                              kernel_size=3, 
                              stride=2,
                              padding=1, 
                              bn_act=True)
        )
        self.conv1 = clayers.StandardConv(self.dims[0]+3, self.dims[0],
                                          kernel_size=3,
                                          stride=2, 
                                          padding=1, 
                                          bn_act=True)
        self.cghost_layer = core.CustomGhostModule(self.dims[0], self.dims[0]//2)
        
        self.downsample_layer2 = nn.Sequential(
            clayers.StandardConv(self.dims[0]*2+3, self.dims[1], 
                                 kernel_size=3, 
                                 stride=2,
                                 padding=1, 
                                 bn_act=False)
        )
        self.downsample_layer3 = nn.Sequential(
            clayers.StandardConv(self.dims[1]*2+3, self.dims[2], 
                                 kernel_size=3, 
                                 stride=2, 
                                 padding=1, 
                                 bn_act=False)
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0

        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        print('LGFI')
                        stage_blocks.append(core.LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    print('asym_dc')
                    stage_blocks.append(
                        core.AsymDilatedConv(in_channels=self.dims[i], 
                                            out_channels=self.asym_dims[i],
                                            dilation=self.dilation[i][j]))
                    
                    # print('CDC')
                    # stage_blocks.append(core.DilatedConv(dim=self.dims[i]//2, k=3, 
                    #                                      dilation=self.dilation[i][j], 
                    #                                      drop_path=dp_rates[cur + j],
                    #                                      layer_scale_init_value=layer_scale_init_value,
                    #                                      expan_ratio=expan_ratio))
            print(' ')
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (clayers.LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # region [forward]
    def forward(self, x):
        x = (x - 0.45) / 0.225
        
        x_down2 = self.avg_pool2(x)
        x_down4 = self.avg_pool4(x)
        x_down8 = self.avg_pool8(x)
        
        """-------------Stage1-----------------"""
        """(48, 96, 320)"""
        stage1_ds2 = self.init_conv(x)
        """(51, 96, 320)"""
        stage1_ds2 = torch.cat((stage1_ds2, x_down2), dim=1)
        """(48, 48, 160)"""
        stage1_ds4 = self.conv1(stage1_ds2)
        # stage1_ds4 = self.dw_conv(stage1_ds2)
        """(48, 48, 160)"""
        stage2_ds4 = self.cghost_layer(stage1_ds4)
        """------------------------------------"""

        """------------Stage2-----------------"""
        # CDC
        for s in range(len(self.stages[0])-1):
            stage2_ds4 = self.stages[0][s](stage2_ds4)  # ch=48
        # LGFI
        stage2_ds4 = self.stages[0][-1](stage2_ds4)     # ch=48
        """-----------------------------------"""
        
        """------------Stage3-----------------"""
        stage3_ds4 = torch.cat([stage1_ds4, stage2_ds4, x_down4], dim=1)  # channel=99
        stage3_ds8 = self.downsample_layer2(stage3_ds4) # channel=80
        # CDC
        for s in range(len(self.stages[1]) - 1):
            stage3_ds8 = self.stages[1][s](stage3_ds8)
        # LGFI
        stage3_ds8_10 = self.stages[1][-1](stage3_ds8) # channel=80
        """-----------------------------------"""
        
        """------------Stage4-----------------"""
        stage4_ds8 = torch.cat([stage3_ds8, stage3_ds8_10, x_down8], dim=1) # channel=163

        stage4_ds16 = self.downsample_layer3(stage4_ds8) # channel=128
        # CDC
        for s in range(len(self.stages[2]) - 1):
            stage4_ds16 = self.stages[2][s](stage4_ds16)
        # LGFI
        stage4_ds16_10 = self.stages[2][-1](stage4_ds16) # channel=128
        """-----------------------------------"""
        
        
        # return features
        return stage2_ds4, stage3_ds8_10, stage4_ds16_10

