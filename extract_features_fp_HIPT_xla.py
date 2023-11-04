import torch
import torch.nn as nn
from math import floor
import os, psutil
import subprocess as sp

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
import vision_transformer as vits
import vision_transformer4k as vits4k
#from hipt_4k import HIPT_4k
import hipt_4k  

from hipt_model_utils import get_vit256, get_vit4k, eval_transforms
import torch_xla_py.xla_model as xm
import torch_xla.distributed.xla_backend
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.distributed.xla_multiprocessing as xmp

#from hipt_heatmap_utils import *

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = xm.xla_device()


print("device ", device)

#device256 = torch.device('cpu')
#device=torch.device('cpu')
device4k=device
device256=device



def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values






def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	print("batch_size ", batch_size)
    
	print("dataset ", len(dataset))
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
    
	print("device ", device)
	print("loader", len(loader))

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	im_features = []
	for count, (batch, coords) in enumerate(loader):

		#print(get_gpu_memory())
		#process = psutil.Process()
		#print(process.memory_info().rss/(1024*1024) )
        
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			#print(batch.shape)
			features = model(batch)
			im_features.append(features.cpu())

			features = features.cpu().numpy()
			print("features.shape",  features.shape)

            #features for imagebind
            
			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'

			x=torch.stack(im_features, dim=0)
	print("im features shape: ", x.shape)
		
	return output_path, x.numpy()


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    

	print('loading model checkpoint')
    


	pretrained_weights4k = "home/shero/Documents/GitHub/HIPT/HIPT_4K/Checkpoints/vit4k_xs_dino.pth"
	pretrained_weights256 = "home/shero/Documents/GitHub/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"
	model = hipt_4k.HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)


	#model = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

	#model = get_vit256(pretrained_weights=pretrained_weights256, device=device256)


    
	#model = resnet50_baseline(pretrained=True)
	#model = model.to(device)
	
	# print_network(model)
    #distributed /// uncomment later 
	#if torch.cuda.device_count() > 1:
		#model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)
	print("total: ",total)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path, im_features = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		print("im features shape: ", im_features.shape)  
		print(bag_name)
		print(bag_base)
		print(args.feat_dir, 'pt_files', bag_base+'.pt')
		#torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
		torch.save(im_features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



