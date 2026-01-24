# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.tcn import TcNetwork



class MultiScale_TemporalConv_1D(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        self.residual = residual
        self.residual_kernel_size = residual_kernel_size

        # 多尺度空洞卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = (effective_kernel_size - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                          padding=padding, dilation=dilation, groups=in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True)
            ))

        # MaxPool分支
        self.maxpool_branch = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels),
            nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 1x1卷积分支（Pointwise）
        self.conv1x1_branch = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)

        # 残差连接
        if self.residual:
            if self.residual_kernel_size == 1:
                self.residual_connection = nn.Identity()
            else:
                padding = (residual_kernel_size - 1) // 2
                self.residual_connection = nn.Sequential(
                    nn.Conv1d(in_channels, in_channels, kernel_size=residual_kernel_size,
                              stride=stride, padding=padding, groups=in_channels),
                    nn.BatchNorm1d(in_channels)
                )

        # 初始化
        self.apply(self.weights_init)

    def forward(self, x):
        # 多尺度空洞卷积分支输出
        out = [branch(x) for branch in self.branches]

        # 池化分支和1x1卷积分支
        out.append(self.maxpool_branch(x))
        out.append(self.conv1x1_branch(x))

        # 融合所有分支（concat → 再平均求和）
        out = torch.stack(out, dim=0).sum(dim=0) / len(out)

        # 残差连接
        if self.residual:
            out += self.residual_connection(x)

        return out

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    
channel_config = [[32,64,128,256],[64,128,256,512],[128,256,512,1024],[256,512,1024,2048]]
config = channel_config[2]

class TSSCD(nn.Module):


    def __init__(self, in_channels = 6, out_channels = 11, config = config):
        super(TSSCD, self).__init__()
        self.out_channels = out_channels
        self.config = config
        c1, c2, c3, c4 = config
        # 第一层卷积
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, 3, padding=1),#B,6,144->B,32,144
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, 3, padding=1),#B,32,144->B,32,144
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/2 #B,32,144->B,32,72
        )
        # 第二层卷积
        self.layer2 = nn.Sequential(
            nn.Conv1d(c1, c2, 3, padding=1), #B,32,72->B,64,72
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c2, 3, padding=1), #B,64,72->B,64,72
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/4 #B,64,72->B,64,36
        )

        # 第三层卷积
        self.layer3 = nn.Sequential(
            nn.Conv1d(c2, c3, 3, padding=1), #B,64,36->B,128,36
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c3, 3, padding=1), #B,128,36->B,128,36
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/8 #B,128,36->B,128,18
        )

        # 第四层卷积
        self.layer4 = nn.Sequential(
            nn.Conv1d(c3, c4, 3, padding=1), #B,128,18->B,256,18
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c4, 3, padding=1), #B,256,18->B,256,18
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/16 #B,256,18->B,256,9
        )      
        
        
        # 第五层使用卷积层取代FC层
        self.score_1 = nn.Conv1d(c4, out_channels, 1) #B,256,9->B,2,9
        self.score_2 = nn.Conv1d(c3, out_channels, 1) #B,128,18->B,2,18
        self.score_3 = nn.Conv1d(c2, out_channels, 1) #B,64,36->B,2,36
        self.score_4 = nn.Conv1d(c1, out_channels, 1) #B,32,72->B,2,72

        # 第六层反卷积
        self.upsampling_2x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,9->B,2,18
        self.upsampling_4x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,18->B,2,36
        self.upsampling_8x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,36->B,2,72
        self.upsampling_16x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,72->B,2,144

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        h = self.layer1(x) #B,6,144->B,32,72
        self.s1 = self.layer2(h) #B,32,72->B,64,36
        self.s2 = self.layer3(self.s1)  # B,64,36->B,128,18
        # B,128,18->B,128,18 #提取更多时间特征 
        
        self.s3 = self.layer4(self.s2)  # B,128,18->B,256,9
        
        s3 = self.score_1(self.s3) # B,256,9->B,2,9
        s3 = self.upsampling_2x(s3) # B,2,9->B,2,18
        s2 = self.score_2(self.s2) # B,128,18->B,2,18

        s2 += s3 # B,2,18
        s2 = self.upsampling_4x(s2) # B,2,18->B,2,36
        s1 = self.score_3(self.s1) # B,64,36->B,2,36
        score = s1 + s2 # B,2,36
        score = self.upsampling_8x(score) # B,2,36->B,2,72
        s0 = self.score_4(h) # B,32,72->B,2,72
        score += s0 # B,2,72
        score = self.upsampling_16x(score) # B,2,72->B,2,144
        return score
    
class TSSCD_D(nn.Module):


    def __init__(self, in_channels = 6, out_channels = 11, config = config):
        super(TSSCD_D, self).__init__()
        self.out_channels = out_channels
        self.config = config
        c1, c2, c3, c4 = config
        # 第一层卷积
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, 3, padding=1),#B,6,144->B,32,144
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, 3, padding=1),#B,32,144->B,32,144
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/2 #B,32,144->B,32,72
        )
        # 第二层卷积
        self.layer2 = nn.Sequential(
            nn.Conv1d(c1, c2, 3, padding=1), #B,32,72->B,64,72
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c2, 3, padding=1), #B,64,72->B,64,72
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/4 #B,64,72->B,64,36
        )

        # 第三层卷积
        self.layer3 = nn.Sequential(
            nn.Conv1d(c2, c3, 3, padding=1), #B,64,36->B,128,36
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c3, 3, padding=1), #B,128,36->B,128,36
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/8 #B,128,36->B,128,18
        )

        # 第四层卷积
        self.layer4 = nn.Sequential(
            nn.Conv1d(c3, c4, 3, padding=1), #B,128,18->B,256,18
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c4, 3, padding=1), #B,256,18->B,256,18
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/16 #B,256,18->B,256,9
        )

        self.MT1 = MultiScale_TemporalConv_1D(in_channels=c3,
                                     kernel_size=3,
                                     stride=1,
                                     dilations=[1, 2, 3, 4],
                                     residual=True,
                                     residual_kernel_size=1)
        self.MT2 = MultiScale_TemporalConv_1D(in_channels=c4,
                                     kernel_size=3,
                                     stride=1,
                                     dilations=[1, 2, 3, 4],
                                     residual=True,
                                     residual_kernel_size=1)
        
        
        # 第六层使用卷积层取代FC层
        self.score_1 = nn.Conv1d(c4, out_channels, 1) #B,256,9->B,2,9
        self.score_2 = nn.Conv1d(c3, out_channels, 1) #B,128,18->B,2,18
        self.score_3 = nn.Conv1d(c2, out_channels, 1) #B,64,36->B,2,36
        self.score_4 = nn.Conv1d(c1, out_channels, 1) #B,32,72->B,2,72

        # 第七层反卷积
        self.upsampling_2x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,9->B,2,18
        self.upsampling_4x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,18->B,2,36
        self.upsampling_8x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,36->B,2,72
        self.upsampling_16x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False) #B,2,72->B,2,144

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        h = self.layer1(x) #B,6,144->B,32,72
        self.s1 = self.layer2(h) #B,32,72->B,64,36
        self.s2 = self.layer3(self.s1)  # B,64,36->B,128,18
        self.s2 = self.MT1(self.s2)

        self.s3 = self.layer4(self.s2)  # B,128,18->B,256,9
        self.s3 = self.MT2(self.s3)

        s3 = self.score_1(self.s3) # B,256,9->B,2,9
        s3 = self.upsampling_2x(s3) # B,2,9->B,2,18
        s2 = self.score_2(self.s2) # B,128,18->B,2,18

        s2 += s3 # B,2,18
        s2 = self.upsampling_4x(s2) # B,2,18->B,2,36
        s1 = self.score_3(self.s1) # B,64,36->B,2,36
        score = s1 + s2 # B,2,36
        score = self.upsampling_8x(score) # B,2,36->B,2,72
        s0 = self.score_4(h) # B,32,72->B,2,72
        score += s0 # B,2,72
        score = self.upsampling_16x(score) # B,2,72->B,2,144
        return score




class TCN(nn.Module):

    
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 11,
        channels: int = 6,
        blocks: int = 4,
        kernel_size: int = 3,
        dilations= 8 ,
    ):
        super().__init__()

        head = {
            "conv_kwargs": [
                {
                    "out_channels": out_channels,  # 最终输出类别数
                    "kernel_size": 1,
                    "dilation": 1,
                    "separable": True,
                }
            ],
        }

        self.tcn = TcNetwork(
            input_channels=6,
            blocks=blocks,
            dilations=dilations,
            channels=channels,
            kernel_size=kernel_size,
            separable=False,
            activation=nn.ReLU,
            top_pooling=head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, T)  e.g. (B, 6, 144)
        return: (B, out_channels, T)
        """
        out_dict = self.tcn(x)          # deep_tcn 返回一个 dict
        out = out_dict["output"]        # 形状 (B, out_channels, T)
        return out
class LSTM(nn.Module):
    """
    输入  (B, in_channels=6, T=144)
    输出  (B, num_classes, T)

    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 11,
        hidden_size: int = 32,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=in_channels,           # 每个时间步的特征维度=6 个波段
            hidden_size=hidden_size,          # LSTM 隐藏维度
            num_layers=num_layers,            # 堆几层 LSTM
            batch_first=True,                 # 输入输出都是 (B, T, C)
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        # 时间步上的逐点分类头：把每个 time step 的特征映射到 num_classes
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)  e.g. (B, 6, 144)

        return:
            logits: (B, num_classes, T)
        """
        # (B, C, T) -> (B, T, C)，适配 batch_first=True 的 LSTM
        if x.shape[1] == self.in_channels:
            x = x.permute(0, 2, 1)  # (B, T, C)

        # LSTM 输出：out: (B, T, lstm_out_dim)
        out, _ = self.lstm(x)

        logits = self.classifier(out)

        logits = logits.permute(0, 2, 1)

        return logits