from .modeling import *
import torch
if torch.__version__ != "0.4.0":
    from ._deeplab import convert_to_separable_conv
