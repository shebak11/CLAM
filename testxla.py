import sys
sys.path.append('/home/MacOS/xla/test')
import args_parse

from torch_xla import runtime as xr
import args_parse

import os
#import schedulers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import torch.distributed as dist
import torch_xla.distributed.xla_backend
import sys
from models.resnet_custom import resnet50_baseline


SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]



MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
    '--test_only_at_end': {
        'action': 'store_true',
    },
    '--ddp': {
        'action': 'store_true',
    },
    '--pjrt_distributed': {
        'action': 'store_true',
    },
    '--profile': {
        'action': 'store_true',
    },
    '--persistent_workers': {
        'action': 'store_true',
    },
    '--prefetch_factor': {
        'type': int,
    },
    '--loader_prefetch_size': {
        'type': int,
    },
    '--device_prefetch_size': {
        'type': int,
    },
    '--host_to_device_transfer_threads': {
        'type': int,
    },
    '--use_optimized_kwargs': {
        'type': str,
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
    #fake_data=True
)


def get_model_property(key):
    default_model_property = {
      'img_dim': 224,
      'model_fn': getattr(torchvision.models, FLAGS.model)
  }
    model_properties = {
      'inception_v3': {
          'img_dim': 299,
          'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
      },
  }
    model_fn = model_properties.get(FLAGS.model, default_model_property)[key]
    return model_fn


def train_imagenet():
    
    print(5)
    
    if FLAGS.ddp or FLAGS.pjrt_distributed:
        dist.init_process_group('xla', init_method='xla://')

    print('==> Preparing data..')
    img_dim = get_model_property('img_dim')
    
    print(7777777777)
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(16, 3, img_dim, img_dim),
              torch.zeros(16, dtype=torch.int64)),
        sample_count=train_dataset_len // 16 //
        xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(16, 3, img_dim, img_dim),
              torch.zeros(16, dtype=torch.int64)),
        sample_count=50000 // 16 // xm.xrt_world_size())

    #if FLAGS.fake_data:

    torch.manual_seed(42)
    device = xm.xla_device()
    print("x")
    model = resnet50_baseline(pretrained=True)
    model = model.to(device)
    #model = get_model_property('model_fn')().to(device)
    
    
    print('y')



def _mp_fn(index, flags):
    #global FLAGS
    #FLAGS = flags
    print(1)
    torch.set_default_dtype(torch.float32)
    #device = xm.xla_device()

    accuracy = train_imagenet()
    print(2)
  #if accuracy < FLAGS.target_accuracy:
    #print('Accuracy {} is below target {}'.format(accuracy,
    #                                              FLAGS.target_accuracy))
    sys.exit(21)

if __name__ == '__main__':
    print(dist.is_torchelastic_launched())
    if dist.is_torchelastic_launched():
        _mp_fn(xu.getenv_as(xenv.LOCAL_RANK, int), FLAGS)
    else:
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
