import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.vos_models.utils import check_parallel, load_checkpoint_epoch,set_stage_stream
from models.vos_models.fusions import GatedConvex, Gated
from models.vos_models.cross_connections import CoAttentionGated, CoAttentionRecip, CoAttentionGatedRecip, CoAttention
import os

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        ######### cross connection types:
        ######### coatt, coatt_gated, coatt_gated_recip, coat_recip, coatt_sum
        if hasattr(args, 'cc_type'):
            self.cc_type = args.cc_type
        else:
            self.cc_type = 'coatt'

        ######## fusion types
        ######## gated, convex gated
        if hasattr(args, 'fusion_type'):
            self.fusion_type = args.fusion_type
        else:
            self.fusion_type = 'gated'

        ######## Normalization Fns
        ######## tanh, softmax
        if hasattr(args, 'norm_fn'):
            norm_fn = args.norm_fn
        else:
            norm_fn = 'tanh'
        print("===> Using Normalization as ", norm_fn)

        ######### Extra Properties
        self.use_spatial_conv = False
        if hasattr(args, 'use_spatial_conv'):
            self.use_spatial_conv = args.use_spatial_conv
        self.use_global = True
        if hasattr(args, 'use_global'):
            self.use_global = args.use_global

        resnet_im = models.resnet101(pretrained=True)
        self.conv1_1 = resnet_im.conv1
        self.bn1_1 = resnet_im.bn1
        self.relu_1 = resnet_im.relu
        self.maxpool_1 = resnet_im.maxpool

        self.res2_1 = resnet_im.layer1
        self.res3_1 = resnet_im.layer2
        self.res4_1 = resnet_im.layer3
        self.res5_1 = resnet_im.layer4

        resnet_fl = models.resnet101(pretrained=True)
        self.conv1_2 = resnet_fl.conv1
        self.bn1_2 = resnet_fl.bn1
        self.relu_2 = resnet_fl.relu
        self.maxpool_2 = resnet_fl.maxpool

        self.res2_2 = resnet_fl.layer1
        self.res3_2 = resnet_fl.layer2
        self.res4_2 = resnet_fl.layer3
        self.res5_2 = resnet_fl.layer4

        kwargs = {}
        if self.fusion_type == 'gated':
            FusionCls = Gated
            kwargs = {'use_global': self.use_global}
        elif self.fusion_type == 'convex_gated':
            FusionCls = GatedConvex
            kwargs = {'use_spatial_conv': self.use_spatial_conv}

        self.gated_res2 = FusionCls(256*2, **kwargs)
        self.gated_res3 = FusionCls(512*2, **kwargs)
        self.gated_res4 = FusionCls(1024*2, **kwargs)
        self.gated_res5 = FusionCls(2048*2, **kwargs)

        if self.cc_type == 'coatt_gated':
            CrossConCls = CoAttentionGated
        elif self.cc_type == 'coatt_gated_recip':
            CrossConCls = CoAttentionGatedRecip
        elif self.cc_type == 'coatt_recip':
            CrossConCls = CoAttentionRecip
        else:
            CrossConCls = CoAttention

        self.coa_res3 = CrossConCls(channel=512, normalization_fn=norm_fn)
        self.coa_res4 = CrossConCls(channel=1024, normalization_fn=norm_fn)
        self.coa_res5 = CrossConCls(channel=2048, normalization_fn=norm_fn)

    def forward_res2(self, f1, f2):
        x1 = self.conv1_1(f1)
        conv1_1 = x1

        x1 = self.bn1_1(x1)
        x1 = self.relu_1(x1)
        x1 = self.maxpool_1(x1)
        r2_1 = self.res2_1(x1)

        x2 = self.conv1_2(f2)
        conv1_2 = x2

        x2 = self.bn1_2(x2)
        x2 = self.relu_2(x2)
        x2 = self.maxpool_2(x2)
        r2_2 = self.res2_2(x2)

        return r2_1, r2_2, conv1_1, conv1_2

    def forward(self, f1, f2, stages):
        interm_out = {}

        requested_stages = []
        requested_streams = []
        for stage in stages:
            stage_, stream_ = stage.split(',')
            requested_stages.append(stage_)
            requested_streams.append(stream_)

        r2_1, r2_2, conv1_1, conv1_2 = self.forward_res2(f1, f2)

        interm_out.update(
            set_stage_stream("conv1", requested_stages, requested_streams, conv1_1, conv1_2, None)
        )

        r2 = torch.cat([r2_1, r2_2], dim=1)
        interm_out.update(
            set_stage_stream("layer1", requested_stages, requested_streams, r2_1, r2_2, None)
        )

        # res3
        r3_1 = self.res3_1(r2_1)
        r3_2 = self.res3_2(r2_2)

        if self.cc_type == 'coatt_add':
            _, Zb, _, Qb = self.coa_res3(r3_1, r3_2)
            r3_1 = F.relu(Zb + r3_1 + r3_2)
            r3_2 = F.relu(Qb + r3_2)
        elif self.cc_type in ['coatt', 'coatt_gated']:
            _, Zb, _, Qb = self.coa_res3(r3_1, r3_2)
            r3_1 = F.relu(Zb + r3_1)
            r3_2 = F.relu(Qb + r3_2)
        elif self.cc_type in ['coatt_recip', 'coatt_gated_recip']:
            Za, Zb, Qa, Qb = self.coa_res3(r3_1, r3_2)
            r3_1 = F.relu(Zb + r3_1)
            r3_2 = F.relu(Za + r3_2)
        else:
            raise NotImplementedError()
        interm_out.update(
            set_stage_stream("layer2", requested_stages, requested_streams, r3_1, r3_2, None)
        )

        # res4
        r4_1 = self.res4_1(r3_1)
        r4_2 = self.res4_2(r3_2)

        if self.cc_type == 'coatt_add':
            _, Zb, _, Qb = self.coa_res4(r4_1, r4_2)
            r4_1 = F.relu(Zb + r4_1 + r4_2)
            r4_2 = F.relu(Qb + r4_2)
        elif self.cc_type in ['coatt', 'coatt_gated']:
            _, Zb, _, Qb = self.coa_res4(r4_1, r4_2)
            r4_1 = F.relu(Zb + r4_1)
            r4_2 = F.relu(Qb + r4_2)
        elif self.cc_type in ['coatt_recip', 'coatt_gated_recip']:
            Za, Zb, Qa, Qb = self.coa_res4(r4_1, r4_2)
            r4_1 = F.relu(Zb + r4_1)
            r4_2 = F.relu(Za + r4_2)
        else:
            raise NotImplementedError()

        interm_out.update(
            set_stage_stream("layer3", requested_stages, requested_streams, r4_1, r4_2, None)
        )


        # res5
        r5_1 = self.res5_1(r4_1)
        r5_2 = self.res5_2(r4_2)

        if self.cc_type == 'coatt_add':
            _, Zb, _, Qb = self.coa_res5(r5_1, r5_2)
            r5_1 = F.relu(Zb + r5_1 + r5_2)
            r5_2 = F.relu(Qb + r5_2)
        elif self.cc_type in ['coatt', 'coatt_gated']:
            _, Zb, _, Qb = self.coa_res5(r5_1, r5_2)
            r5_1 = F.relu(Zb + r5_1)
            r5_2 = F.relu(Qb + r5_2)
        elif self.cc_type in ['coatt_recip', 'coatt_gated_recip']:
            Za, Zb, Qa, Qb = self.coa_res5(r5_1, r5_2)
            r5_1 = F.relu(Zb + r5_1)
            r5_2 = F.relu(Za + r5_2)
        else:
            raise NotImplementedError()
        interm_out.update(
            set_stage_stream("layer4", requested_stages, requested_streams, r5_1, r5_2, None)
        )

        if self.fusion_type == 'convex_gated':
            r5_gated = self.gated_res5(r5_1, r5_2)
            r4_gated = self.gated_res4(r4_1, r4_2)
            r3_gated = self.gated_res3(r3_1, r3_2)
            r2_gated = self.gated_res2(r2_1, r2_2)

        elif self.fusion_type == 'gated':
            r2 = torch.cat([r2_1, r2_2], dim=1)
            r3 = torch.cat([r3_1, r3_2], dim=1)
            r4 = torch.cat([r4_1, r4_2], dim=1)
            r5 = torch.cat([r5_1, r5_2], dim=1)

            r5_gated = self.gated_res5(r5)
            r4_gated = self.gated_res4(r4)
            r3_gated = self.gated_res3(r3)
            r2_gated = self.gated_res2(r2)
        else:
            raise NotImplementedError()

        interm_out.update(
            set_stage_stream("layer4", requested_stages, requested_streams, None, None, r5_gated)
        )
        interm_out.update(
            set_stage_stream("layer3", requested_stages, requested_streams, None, None, r4_gated)
        )
        interm_out.update(
            set_stage_stream("layer2", requested_stages, requested_streams, None, None, r3_gated)
        )
        interm_out.update(
            set_stage_stream("layer1", requested_stages, requested_streams, None, None, r2_gated)
        )
        return r5_gated, r4_gated, r3_gated, r2_gated, interm_out


class BoundaryModule(nn.Module):
    def __init__(self, inchannel):
        super(BoundaryModule, self).__init__()

        self.bn1 = nn.BatchNorm2d(inchannel)
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1,
                               padding=1)
        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        mdim = 256
        self.GC = GC(4096+1, mdim)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(2048+1, mdim)
        self.RF3 = Refine(1024+1, mdim)
        self.RF2 = Refine(512+1, mdim)

        self.pred5 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.concat = nn.Conv2d(4, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.bdry5 = BoundaryModule(4096)
        self.bdry4 = BoundaryModule(2048)
        self.bdry3 = BoundaryModule(1024)
        self.bdry2 = BoundaryModule(512)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, r5, r4, r3, r2):
        p5 = self.bdry5(r5)
        p4 = self.bdry4(r4)
        p3 = self.bdry3(r3)
        p2 = self.bdry2(r2)

        p2_up = F.interpolate(p2, size=(473, 473), mode='bilinear')
        p3_up = F.interpolate(p3, size=(473, 473), mode='bilinear')
        p4_up = F.interpolate(p4, size=(473, 473), mode='bilinear')
        p5_up = F.interpolate(p5, size=(473, 473), mode='bilinear')

        concat = torch.cat([p2_up, p3_up, p4_up, p5_up], dim=1)
        p = self.concat(concat)

        p2_up = torch.sigmoid(p2_up)
        p3_up = torch.sigmoid(p3_up)
        p4_up = torch.sigmoid(p4_up)
        p5_up = torch.sigmoid(p5_up)
        p = torch.sigmoid(p)

        r5 = torch.cat((r5, p5), dim=1)
        r4 = torch.cat((r4, p4), dim=1)
        r3 = torch.cat((r3, p3), dim=1)
        r2 = torch.cat((r2, p2), dim=1)

        m, final_f = self.forward_mask(r5, r4, r3, r2)

        return m, p, p2_up, p3_up, p4_up, p5_up, final_f

    def forward_mask(self, x, r4, r3, r2):
        x = self.GC(x)
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
        m5 = x + r
        m4 = self.RF4(r4, m5)
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2))
        p2_up = F.interpolate(p2, size=(473, 473), mode='bilinear')
        p2_s = torch.sigmoid(p2_up)

        return p2_s, m2


class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class AtrousBlock(nn.Module):
    def __init__(self, inplanes, planes, rate, stride=1):
        super(AtrousBlock, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              dilation=rate, padding=rate)

    def forward(self, x):
        return self.conv(x)


class PyramidDilationConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(PyramidDilationConv, self).__init__()

        rate = [3, 5, 7]

        self.block0 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.block1 = AtrousBlock(inplanes, planes, rate[0])
        self.block2 = AtrousBlock(inplanes, planes, rate[1])
        self.block3 = AtrousBlock(inplanes, planes, rate[2])
        self.bn = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block1(x)
        x3 = self.block1(x)

        xx = torch.cat([x0, x1, x2, x3], dim=1)
        xx = self.bn(xx)
        return xx


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

        outplanes = int(planes / 4)
        self.pdc = PyramidDilationConv(inplanes, outplanes)

    def forward(self, f, pm):
        s = self.pdc(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        m = s + F.interpolate(pm, size=s.shape[2:4], mode='bilinear')

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m

class MATNet(nn.Module):
    def __init__(self, args):
        super(MATNet, self).__init__()

        self.encoder = Encoder(args=args)
        self.decoder = Decoder()

    def forward(self, x, stage):
        r5, r4, r3, r2, interm_out = self.encoder(x['Image'], x['Flow'], stage)
        out, _, _, _, _, _, layerf = self.decoder(r5, r4, r3, r2)
        interm_out['layerf,sensor_fusion'] = layerf.view(*layerf.shape[:2], -1).mean(dim=2).detach().cpu().numpy()
        return out, interm_out

    def load_ckpt(self, checkpoint):
        ckpts = os.listdir(checkpoint)
        last_epoch = 0
        for ckpt in ckpts:
            if 'encoder' in ckpt and '_' in ckpt:
                current_epoch = int(ckpt.split('.')[0].split('_')[1])
                if current_epoch > last_epoch:
                    last_epoch = current_epoch

        print('Loading Epoch ', last_epoch)
        encoder_dict, decoder_dict, _, _, _ =\
            load_checkpoint_epoch(checkpoint, last_epoch, True, False)
        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        # Since some layers are not being used by MATNet model and are excluded in my training
        # setting stric to be False to allow loading mine and author's models
        self.encoder.load_state_dict(encoder_dict, strict=False)
        self.decoder.load_state_dict(decoder_dict, strict=False)

