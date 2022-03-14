import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Gated(nn.Module):
    def __init__(self, channel, reduction=16, use_global=True):
        super(Gated, self).__init__()

        self.use_global = use_global
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.excitation_1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True))

        self.excitation_2 = nn.Sequential(
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

        self.global_attention = nn.Sequential(
            nn.Linear(channel // reduction, 1),
            nn.Sigmoid()
        )

        kernel_size = 7
        self.spatial = BasicConv(1, 1, kernel_size, stride=1,
                                 padding=(kernel_size-1) // 2, relu=False)
        print("====> Creating Gated Fusion")

    def forward(self, U):
        # se layer
        b, c, h, w = U.shape
        S = self.avg_pool(U).view(b, c)
        E_1 = self.excitation_1(S)

        E_local = self.excitation_2(E_1).view(b, c, 1, 1)
        U_se = E_local * U

        # spatial layer
        U_se_max = torch.max(U_se, 1)[0].unsqueeze(1)
        SP_Att = self.spatial(U_se_max)
        U_se_sp = SP_Att * U_se

        # global layer
        if self.use_global:
            E_global = self.global_attention(E_1).view(b, 1, 1, 1)
            V = E_global * U_se_sp
        else:
            V = U_se_sp

        # residual layer
        O = U + V

        return O

#############################################################################

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, use_spatial_conv=False):
        super(SpatialAttention, self).__init__()
        self.use_spatial_conv = use_spatial_conv

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        if self.use_spatial_conv:
            kernel_size = 7
            self.conv1 = BasicConv(1, 1, kernel_size, stride=1,
                                   padding=(kernel_size-1) // 2,
                                   relu=False, bn=False)
        else:
            self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, flow):
        if self.use_spatial_conv:
            x = torch.max(torch.cat([img, flow], dim=1), 1)[0].unsqueeze(1)
            x = self.sigmoid(self.conv1(x))
        else:
            x = torch.cat([torch.mean(img, dim=1, keepdim=True),
                           torch.max(img, dim=1, keepdim=True)[0],
                           torch.mean(flow, dim=1, keepdim=True),
                           torch.max(flow, dim=1, keepdim=True)[0]], dim=1)
            x = self.sigmoid(self.conv1(x))
        return torch.cat([img.mul(x), flow.mul(1-x)], dim=1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes // 2, 1)
        self.in_planes=in_planes

    def forward(self, img, flow):
        f = self.fc2(self.relu1(self.fc1(self.avg_pool(torch.cat([img, flow], 1)))))#B 2C 1 1
        f = F.sigmoid(f)
        return img.mul(f), flow.mul(1-f)

class GatedConvex(nn.Module):
    def __init__(self, inplanes, ratio=16, use_spatial_conv=False):
        super(GatedConvex, self).__init__()
        self.channel = ChannelAttention(inplanes, ratio=ratio)
        self.spatial = SpatialAttention(use_spatial_conv=use_spatial_conv)
        print("====> Creating Convex Gated Fusion")

    def forward(self, spatial, temporal):
        spatial, temporal = self.channel(spatial, temporal)
        feature = self.spatial(spatial, temporal)
        return feature

