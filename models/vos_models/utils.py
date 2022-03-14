import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import os

def set_stage_stream(current_stage, requested_stages, requested_streams, app, mot, sensorfusion):
    interm_out = {}
    if current_stage in requested_stages:
        indices = [i for i, st in enumerate(requested_stages) if current_stage==st]
        for idx in indices:
            if requested_streams[idx] == "app_stream" and app is not None:
                interm_out["%s,%s"%(current_stage, requested_streams[idx])] = \
                        app.view(app.shape[0], app.shape[1], -1).mean(dim=2).detach().cpu().numpy()

            if requested_streams[idx] == "mot_stream" and mot is not None:
                interm_out["%s,%s"%(current_stage, requested_streams[idx])] = \
                        mot.view(mot.shape[0], mot.shape[1], -1).mean(dim=2).detach().cpu().numpy()

            if requested_streams[idx] == "sensor_fusion" and sensorfusion is not None:
                interm_out["%s,%s"%(current_stage, requested_streams[idx])] = \
                      sensorfusion.view(sensorfusion.shape[0], sensorfusion.shape[1], -1).mean(dim=2).detach().cpu().numpy()

    return interm_out

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, stage=None):
        if type(x) == dict:
            input_shape = x['Image'].shape[-2:]
        else:
            input_shape = x.shape[-2:]

        features, interm_out = self.backbone(x, stage)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x, interm_out


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def load_checkpoint_epoch(model_name, epoch, use_gpu=True, load_opt=True):
    encoder_dict = torch.load(os.path.join(model_name, 'encoder_{}.pt'.format(epoch)))
    decoder_dict = torch.load(os.path.join(model_name, 'decoder_{}.pt'.format(epoch)))
    return encoder_dict, decoder_dict, None, None, None

def check_parallel(encoder_dict, decoder_dict):
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict


