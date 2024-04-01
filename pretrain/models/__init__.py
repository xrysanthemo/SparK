# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from timm import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.layers import drop

from models.spark_convnext import ConvNeXt
from models.spark_resnet import ResNet
from models.spark_custom import YourConvNet
_import_resnets_for_timm_registration = (ResNet,)


# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, drop.DropPath):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


pretrain_default_model_kwargs = {
    'your_convnet': dict(),
    'resnet18': dict(),
    'resnet50': dict(drop_path_rate=0.05),
    'resnet101': dict(drop_path_rate=0.08),
    'resnet152': dict(drop_path_rate=0.10),
    'resnet200': dict(drop_path_rate=0.15),
    'convnext_small': dict(sparse=True, drop_path_rate=0.2),
    'convnext_base': dict(sparse=True, drop_path_rate=0.3),
    'convnext_large': dict(sparse=True, drop_path_rate=0.4),
}
for kw in pretrain_default_model_kwargs.values():
    kw['pretrained'] = False
    kw['num_classes'] = 0
    kw['global_pool'] = ''




