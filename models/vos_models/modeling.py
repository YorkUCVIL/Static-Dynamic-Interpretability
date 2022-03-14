import torch
from easydict import EasyDict as edict
if torch.__version__ == "0.4.0":
    from .segment_any_moving_interface import SAM
    def sam(num_classes=21, output_stride=8, pretrained_backbone=True, fuse_early=True,
               pretrain_motionstream=False, fuse_bnorm=False):
        """
        Create Interface to Segment Any Moving model
        """
        model = SAM()
        return model

else:
    from .utils import IntermediateLayerGetter
    from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
    from .backbone import resnet
    from .backbone import mobilenetv2
    from .backbone import twostream
    from .MATNet import MATNet
    from .rtnet.interactive import Interactive
    from .rtnet_34.interactive import Interactive as InteractiveR34

    def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone,
                     fuse_early=True, pretrain_motionstream=False, fuse_bnorm=False):

        if output_stride==8:
            replace_stride_with_dilation=[False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation=[False, False, True]
            aspp_dilate = [6, 12, 18]

        ### Construct Segmentation Layer
        inplanes = 2048
        low_level_planes = 256

        if name=='deeplabv3plus':
            return_layers = {'layer4': 'out', 'layer1': 'low_level'}
            classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate,
                                            fuse_early=fuse_early)
        elif name=='deeplabv3':
            return_layers = {'layer4': 'out'}
            classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)

        ### Construct Backbone
        if 'twostream' in backbone_name:
            backbone = twostream.TwoStream(backbone_name, pretrained_backbone,
                                 replace_stride_with_dilation, return_layers,
                                 fuse_early=fuse_early, pretrain_motionstream=pretrain_motionstream,
                                 fuse_bnorm=fuse_bnorm)
        else:
            backbone = resnet.__dict__[backbone_name](
                pretrained=pretrained_backbone,
                replace_stride_with_dilation=replace_stride_with_dilation)
            backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        model = DeepLabV3(backbone, classifier)
        return model

    def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
        if output_stride==8:
            aspp_dilate = [12, 24, 36]
        else:
            aspp_dilate = [6, 12, 18]

        backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

        # rename layers
        backbone.low_level_features = backbone.features[0:4]
        backbone.high_level_features = backbone.features[4:-1]
        backbone.features = None
        backbone.classifier = None

        inplanes = 320
        low_level_planes = 24

        if name=='deeplabv3plus':
            return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
            classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        elif name=='deeplabv3':
            return_layers = {'high_level_features': 'out'}
            classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        model = DeepLabV3(backbone, classifier)
        return model

    def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone,
                    fuse_early=True, pretrain_motionstream=False, fuse_bnorm=False):

        if backbone=='mobilenetv2':
            model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
        elif backbone.startswith('resnet'):
            model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                                 pretrained_backbone=pretrained_backbone, fuse_early=fuse_early,
                                 pretrain_motionstream=pretrain_motionstream, fuse_bnorm=fuse_bnorm)
        else:
            raise NotImplementedError
        return model

    # Deeplab v3
    def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True,
                           fuse_early=False, pretrain_motionstream=False):
        """Constructs a DeepLabV3 model with a ResNet-50 backbone.

        Args:
            num_classes (int): number of classes.
            output_stride (int): output stride for deeplab.
            pretrained_backbone (bool): If True, use the pretrained backbone.
        """
        return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride,
                            pretrained_backbone=pretrained_backbone)

    def twostream_deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True,
                                          fuse_early=True, pretrain_motionstream=False, fuse_bnorm=False,
                                          extra_args=None):
        """Constructs a DeepLabV3 model with a ResNet-101 backbone.

        Args:
            num_classes (int): number of classes.
            output_stride (int): output stride for deeplab.
            pretrained_backbone (bool): If True, use the pretrained backbone.
        """
        return _load_model('deeplabv3plus', 'resnet101_twostream', num_classes, output_stride=output_stride,
                            pretrained_backbone=pretrained_backbone, fuse_early=fuse_early,
                            pretrain_motionstream=pretrain_motionstream, fuse_bnorm=fuse_bnorm)

    def matnet(num_classes=21, output_stride=8, pretrained_backbone=True, fuse_early=True,
               pretrain_motionstream=False, fuse_bnorm=False, extra_args=None):
        """
        Create Motion Attentive network
        """
        matnet_args = edict(extra_args)
        matnet = MATNet(matnet_args)
        return matnet


    def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True,
                            fuse_early=False, pretrain_motionstream=False):
        """Constructs a DeepLabV3 model with a ResNet-101 backbone.

        Args:
            num_classes (int): number of classes.
            output_stride (int): output stride for deeplab.
            pretrained_backbone (bool): If True, use the pretrained backbone.
        """
        return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True,
                            fuse_early=False, pretrain_motionstream=False, **kwargs):
        """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

        Args:
            num_classes (int): number of classes.
            output_stride (int): output stride for deeplab.
            pretrained_backbone (bool): If True, use the pretrained backbone.
        """
        return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


    # Deeplab v3+
    def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True,
                               fuse_early=True, pretrain_motionstream=False):
        """Constructs a DeepLabV3 model with a ResNet-50 backbone.

        Args:
            num_classes (int): number of classes.
            output_stride (int): output stride for deeplab.
            pretrained_backbone (bool): If True, use the pretrained backbone.
        """
        return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


    def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True,
                               fuse_early=True, pretrain_motionstream=False):
        """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

        Args:
            num_classes (int): number of classes.
            output_stride (int): output stride for deeplab.
            pretrained_backbone (bool): If True, use the pretrained backbone.
        """
        return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                            pretrained_backbone=pretrained_backbone)


    def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True,
                               fuse_early=True, pretrain_motionstream=False):
        """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

        Args:
            num_classes (int): number of classes.
            output_stride (int): output stride for deeplab.
            pretrained_backbone (bool): If True, use the pretrained backbone.
        """
        return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    def rtnet(num_classes=21, output_stride=8, pretrained_backbone=True, fuse_early=True,
               pretrain_motionstream=False, fuse_bnorm=False, extra_args=None):
        """
        Create Motion Attentive network
        """
        rtnet = Interactive()
        return rtnet

    def rtnet34(num_classes=21, output_stride=8, pretrained_backbone=True, fuse_early=True,
                pretrain_motionstream=False, fuse_bnorm=False):
        """
        Create Motion Attentive network
        """
        rtnet = InteractiveR34()
        return rtnet
