from torch_xla import runtime as xr
import sys
sys.path.append('/home/MacOS/xla/test')
import args_parse
from models.resnet_custom import resnet50_baseline


import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
#import tiffslide as openslide
from tiffslide import TiffSlide
from google.cloud import storage
from multiprocessing import Manager
import pickle 
from datetime import datetime



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
)

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

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
    persistent_workers=False,
    prefetch_factor=16,
    loader_prefetch_size=8,
    device_prefetch_size=4,
    num_workers=8,
    host_to_device_transfer_threads=1,
)

#  Best config to achieve peak performance based on TPU version
#    1. It is recommended to use this config in conjuntion with XLA_USE_BF16=1 Flag.
#    2. Hyperparameters can be tuned to further improve the accuracy.
#  usage: python3 /usr/share/pytorch/xla/test/test_train_mp_imagenet.py --model=resnet50 \
#         --fake_data --num_epochs=10 --log_steps=300 \
#         --profile   --use_optimized_kwargs=tpuv4  --drop_last
OPTIMIZED_KWARGS = {
    'tpuv4':
        dict(
            batch_size=128,
            test_set_batch_size=128,
            num_epochs=18,
            momentum=0.9,
            lr=0.1,
            target_accuracy=0.0,
            persistent_workers=True,
            prefetch_factor=32,
            loader_prefetch_size=128,
            device_prefetch_size=1,
            num_workers=16,
            host_to_device_transfer_threads=4,
        )
}

MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS/OPTIMIZED_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            OPTIMIZED_KWARGS.get(FLAGS.use_optimized_kwargs, DEFAULT_KWARGS),
            **{
                'lr': 0.5,
                'lr_scheduler_divide_every_n_epochs': 20,
                'lr_scheduler_divisor': 5,
                'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
            })
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)


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


def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_imagenet(index=0):
  if FLAGS.ddp or FLAGS.pjrt_distributed:
    dist.init_process_group('xla', init_method='xla://')
  data_h5_dir = "WSI/TCGA/COADtest_dir/patches/" 
  data_slide_dir  = "WSI/TCGA/COAD/" 
  csv_path = "WSI/TCGA/COADtest_dir/process_list_autogen.csv" 
  feat_dir = "WSI/TCGA/COADtest_features_dir/" 
  #batch_size = 8 
  slide_ext = ".svs"    
  csv_path = "WSI/TCGA/COADtest_dir/process_list_autogen.csv" 
  bags_dataset = Dataset_All_Bags(csv_path)
    
  verbose = 1
  print_every=20
  pretrained=True 
  custom_downsample=1
  target_patch_size=224

  total = len(bags_dataset)
  for bag_candidate_idx in range(2):
      slide_id = bags_dataset[bag_candidate_idx].split(slide_ext)[0]
      file_id = os.path.basename(slide_id)
      bag_name = os.path.basename(slide_id)+'.h5'
      gs_file_path = os.path.join(data_h5_dir, bag_name)
      gs_slide_file_path = os.path.join(data_slide_dir, file_id+slide_ext)
      local_slide_file_path = "/home/MacOS/"+ file_id+slide_ext
      local_file_path = "/home/MacOS/"+bag_name
      local_ofile_path = "/home/MacOS/" + "h5_files/" +str(index)+"_" + bag_name
      gs_ofile_path = os.path.join(feat_dir, "h5_files2/" + str(index)+"_" +bag_name)
    
      storage_client = storage.Client()
      bucket = storage_client.bucket("oncomerge")
      blob = bucket.blob(gs_file_path)
      with blob.open("rb") as f:
        with h5py.File(f,'r') as hdf5_file:
            coord = hdf5_file['coords'][0]
      print("coord")
      print(type(coord))
      print(coord) 
      print(coord.shape)
      print(type(coord[0]))
      print('==> Preparing data..')
      img_dim = get_model_property('img_dim')

      if FLAGS.fake_data:
        print(FLAGS.num_workers) #4
        print(FLAGS.persistent_workers) #false
        print(FLAGS.prefetch_factor) #16
        #quit()
        train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
        train_loader = xu.SampleGenerator(
            data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
                  torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
            sample_count=train_dataset_len // FLAGS.batch_size //
            xm.xrt_world_size())
        test_loader = xu.SampleGenerator(
            data=(torch.zeros(FLAGS.test_set_batch_size, 3, img_dim, img_dim),
                  torch.zeros(FLAGS.test_set_batch_size, dtype=torch.int64)),
            sample_count=50000 // FLAGS.batch_size // xm.xrt_world_size())
        dataset = Whole_Slide_Bag_FP(file_path=gs_file_path, gs_slide_file_path=gs_slide_file_path, pretrained=pretrained,  custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        print(len(dataset))
        print(np.array(dataset[0][0]).shape)
        print((dataset[0][1]))
        test_sampler = None, None
        
        my_test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.test_set_batch_size,
            ##sampler=test_sampler,
            ##drop_last=FLAGS.drop_last,
            ##shuffle=False,
            #num_workers=FLAGS.num_workers,
            #persistent_workers=FLAGS.persistent_workers,
            #prefetch_factor=FLAGS.prefetch_factor,
            collate_fn=collate_features)
        
      else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.datadir, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(img_dim),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_dataset_len = len(train_dataset.imgs)
        resize_dim = max(img_dim, 256)
        test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.datadir, 'val'),
            # Matches Torchvision's eval transforms except Torchvision uses size
            # 256 resize for all models both here and in the train loader. Their
            # version crashes during training on 299x299 images, e.g. inception.
            transforms.Compose([
                transforms.Resize(resize_dim),
                transforms.CenterCrop(img_dim),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler, test_sampler = None, None
        if xm.xrt_world_size() > 1:
          train_sampler = torch.utils.data.distributed.DistributedSampler(
              train_dataset,
              num_replicas=xm.xrt_world_size(),
              rank=xm.get_ordinal(),
              shuffle=True)
          test_sampler = torch.utils.data.distributed.DistributedSampler(
              test_dataset,
              num_replicas=xm.xrt_world_size(),
              rank=xm.get_ordinal(),
              shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            sampler=train_sampler,
            drop_last=FLAGS.drop_last,
            shuffle=False if train_sampler else True,
            num_workers=FLAGS.num_workers,
            persistent_workers=FLAGS.persistent_workers,
            prefetch_factor=FLAGS.prefetch_factor)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FLAGS.test_set_batch_size,
            sampler=test_sampler,
            drop_last=FLAGS.drop_last,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            persistent_workers=FLAGS.persistent_workers,
            prefetch_factor=FLAGS.prefetch_factor)

      torch.manual_seed(42)

      device = xm.xla_device()
      #model = get_model_property('model_fn')().to(device)
      model = resnet50_baseline(pretrained=True)
      model = model.to(device)

      # Initialization is nondeterministic with multiple threads in PjRt.
      # Synchronize model parameters across replicas manually.
      if xr.using_pjrt():
        xm.broadcast_master_param(model)

      if FLAGS.ddp:
        model = DDP(model, gradient_as_bucket_view=True, broadcast_buffers=False)

      writer = None
      if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
      optimizer = optim.SGD(
          model.parameters(),
          lr=FLAGS.lr,
          momentum=FLAGS.momentum,
          weight_decay=1e-4)
      num_training_steps_per_epoch = train_dataset_len // (
          FLAGS.batch_size * xm.xrt_world_size())
      lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
          optimizer,
          scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
          scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
          scheduler_divide_every_n_epochs=getattr(
              FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
          num_steps_per_epoch=num_training_steps_per_epoch,
          summary_writer=writer)
      loss_fn = nn.CrossEntropyLoss()

      if FLAGS.profile:
        server = xp.start_server(FLAGS.profiler_port)

      def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()
        for step, (data, target) in enumerate(loader):
          with xp.StepTrace('train_imagenet'):
            with xp.Trace('build_graph'):
              optimizer.zero_grad()
              output = model(data)
              loss = loss_fn(output, target)
              loss.backward()
              if FLAGS.ddp:
                optimizer.step()
              else:
                xm.optimizer_step(optimizer)
                tracker.add(FLAGS.batch_size)
              if lr_scheduler:
                lr_scheduler.step()
            if step % FLAGS.log_steps == 0:
              xm.add_step_closure(
                  _train_update, args=(device, step, loss, tracker, epoch, writer))

      def test_loop_fn(loader, epoch, local_ofile_path, gs_ofile_path):
        total_samples, correct = 0, 0
        model.eval()
        mode = 'w'
        for step, (data, target) in enumerate(loader):
          output = model(data)
          features = output.detach().cpu().numpy()
          #asset_dict = {'features': features, 'coords': coords}
          asset_dict = {'features': features} 
          save_hdf5(local_ofile_path, asset_dict, attr_dict= None, mode=mode)
          mode = 'a'
          pred = output.max(1, keepdim=True)[1]
          correct += pred.eq(target.view_as(pred)).sum()
          total_samples += data.size()[0]
          if step % FLAGS.log_steps == 0:
            xm.add_step_closure(
                test_utils.print_test_update, args=(device, None, epoch, step))
        accuracy = 100.0 * correct.item() / total_samples
        accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
        storage_client = storage.Client()
        bucket = storage_client.bucket("oncomerge")
        stats = storage.Blob(bucket=bucket, name=gs_ofile_path).exists(storage_client)
        if not stats:
          blob = bucket.blob(gs_ofile_path)
          blob.upload_from_filename(local_ofile_path )
          print(local_ofile_path)
        return accuracy

      train_device_loader = pl.MpDeviceLoader(
          train_loader,
          device,
          loader_prefetch_size=FLAGS.loader_prefetch_size,
          device_prefetch_size=FLAGS.device_prefetch_size,
          host_to_device_transfer_threads=FLAGS.host_to_device_transfer_threads)
      test_device_loader = pl.MpDeviceLoader(
          test_loader,
          device,
          loader_prefetch_size=FLAGS.loader_prefetch_size,
          device_prefetch_size=FLAGS.device_prefetch_size,
          host_to_device_transfer_threads=FLAGS.host_to_device_transfer_threads)
      accuracy, max_accuracy = 0.0, 0.0
      #print(len(test_loader))
      print(xm.xrt_world_size())
      print((FLAGS.test_set_batch_size))
      print(len(test_device_loader))
      print(len(test_loader))
      #(batch, coords) = next(iter(data_loader)) 
      temp = next(iter(test_loader)) 
      print(len(temp))
      print(FLAGS.num_workers)
      #quit()
      for epoch in range(1, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        #train_loop_fn(train_device_loader, epoch)
        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
        if not FLAGS.test_only_at_end or epoch == FLAGS.num_epochs:
          accuracy = test_loop_fn(test_device_loader, epoch, local_ofile_path, gs_ofile_path)
          xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
              epoch, test_utils.now(), accuracy))
          max_accuracy = max(accuracy, max_accuracy)
          test_utils.write_to_summary(
              writer,
              epoch,
              dict_to_write={'Accuracy/test': accuracy},
              write_xla_metrics=True)
        if FLAGS.metrics_debug:
          xm.master_print(met.metrics_report())

      test_utils.close_summary_writer(writer)
      xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_dtype(torch.float32)
  accuracy = train_imagenet(index)
  if accuracy < FLAGS.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
  if dist.is_torchelastic_launched():
    _mp_fn(xu.getenv_as(xenv.LOCAL_RANK, int), FLAGS)
  else:
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)