import torch
import torch.nn as nn
from . import resnet
from ..utils import IntermediateLayerGetter
from collections import OrderedDict
from models.vos_models.utils import set_stage_stream

class TwoStream(nn.Module):
    def __init__(self, backbone_name, pretrained_backbone,
            replace_stride_with_dilation, return_layers, fuse_early=True,
            pretrain_motionstream=False, fuse_bnorm=False):
        super(TwoStream, self).__init__()

        backbone_name = backbone_name.replace('_twostream', '')

        self.app_stream = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)
        self.app_stream = IntermediateLayerGetter(self.app_stream, return_layers=return_layers)

        self.mot_stream = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)
        self.mot_stream = IntermediateLayerGetter(self.mot_stream, return_layers=return_layers)

        # Pretrain app_stream
        if pretrained_backbone:
            state_dict = torch.load(
                'checkpoints/best_deeplabv3plus_resnet101_voc_os16.pth')

            app_stream_state_dict = self.app_stream.state_dict()

            for name, weights in state_dict["model_state"].items():
                if 'backbone' in name:
                    name = name.replace('backbone.', '')
                    app_stream_state_dict[name] = weights

            self.app_stream.load_state_dict(app_stream_state_dict)
            print("Loaded Appearance Stream with Pretrained Pascal Weights")

        if pretrain_motionstream:
            state_dict = torch.load(
                'checkpoints/latest_deeplabv3plus_resnet101_taovos_os16.pth')

            mot_stream_state_dict = self.mot_stream.state_dict()
            for name, weights in state_dict["model_state"].items():
                if 'backbone' in name:
                    name = name.replace('backbone.', '')
                    mot_stream_state_dict[name] = weights

            self.mot_stream.load_state_dict(mot_stream_state_dict)
            print("Loaded Motion Stream with Pretrained TAO-VOS Weights")

        self.return_layers = return_layers
        self.fuse_early = fuse_early
        self.fuse_bnorm = fuse_bnorm

        if self.fuse_early:
            if self.fuse_bnorm:
                self.sensor_fusion = nn.ModuleDict({'layer1': nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256)),
                                                    'layer2': nn.Sequential(nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512)),
                                                    'layer3': nn.Sequential(nn.Conv2d(2048, 1024, 1), nn.BatchNorm2d(1024)),
                                                    'layer4': nn.Sequential(nn.Conv2d(4096, 2048, 1), nn.BatchNorm2d(2048)) })
            else:
                self.sensor_fusion = nn.ModuleDict({'layer1': nn.Conv2d(512, 256, 1),
                                      'layer2': nn.Conv2d(1024, 512, 1),
                                      'layer3': nn.Conv2d(2048, 1024, 1),
                                      'layer4': nn.Conv2d(4096, 2048, 1)})


    def forward(self, x, stages):
        y = x['Flow']
        x = x['Image']
        interm_out = {}

        requested_stages = []
        requested_streams = []
        for stage in stages:
            stage_, stream_ = stage.split(',')
            requested_stages.append(stage_)
            requested_streams.append(stream_)

        out = OrderedDict()
        for (name, app_module), (_, mot_module) in \
                zip(self.app_stream.named_children(), self.mot_stream.named_children()):
            x = app_module(x)
            y = mot_module(y)

            if name in ['bn1', 'relu', 'maxpool']:
                continue

            interm_out.update(
                set_stage_stream(name, requested_stages, requested_streams, x, y, None)
            )

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if self.fuse_early:
                    out[out_name] = self.sensor_fusion[name](
                            torch.cat((x, y), dim=1) )
                else:
                    out[out_name] = [x, y]

                interm_out.update(
                    set_stage_stream(name, requested_stages, requested_streams, None, None, out[out_name])
                )
        return out, interm_out
