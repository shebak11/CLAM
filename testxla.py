from torch_xla import runtime as xr
#import args_parse

import os
import schedulers
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


if __name__ == '__main__':
    print(dist.is_torchelastic_launched())
  #if dist.is_torchelastic_launched():
    #_mp_fn(xu.getenv_as(xenv.LOCAL_RANK, int), FLAGS)
  #else:
    #xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
