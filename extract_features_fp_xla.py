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
from google.cloud import storage

from torch_xla import runtime as xr

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}

	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
	print("len(loader)")
	print(len(loader))
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		if count==25:
			break
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			local_output_path = "/home/MacOS/h5_files/"+os.path.basename(output_path)
			print("local_output_path" + local_output_path)
			save_hdf5(local_output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	storage_client = storage.Client()
	bucket = storage_client.bucket("oncomerge")
	blob = bucket.blob(output_path)
	blob.upload_from_filename(local_output_path )
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')  
	#if FLAGS.ddp or FLAGS.pjrt_distributed:
		#dist.init_process_group('xla', init_method='xla://')  
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	#os.makedirs(args.feat_dir, exist_ok=True)
	#os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	#os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	#dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
	dest_files=[]
	x = 5
	storage_client = storage.Client()
	bucket = storage_client.bucket("oncomerge")
	print("")
	blobs = storage_client.list_blobs("oncomerge", prefix=args.feat_dir+'pt_files/')
	for blob in blobs:
		dest_files.append(blob.name)

	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)
	total = 3
	print( "len(bags_dataset)")
	print( len(bags_dataset))

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = os.path.basename(slide_id)+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		#storage_client = storage.Client()
		path = args.data_slide_dir+os.path.basename(slide_id)+args.slide_ext    
		blob = bucket.blob(path)
		print(os.path.basename(slide_id))
		slide_file_path = "/home/MacOS/"+ os.path.basename(slide_id)+args.slide_ext  
		print( "slide_file_path " + slide_file_path)
		blob.download_to_filename(slide_file_path )
		#self.wsi = openslide.OpenSlide(path) 
		wsi = openslide.open_slide(slide_file_path)
		#wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		local_output_path = "/home/MacOS/h5_files/"+os.path.basename(output_path)
		file = h5py.File(local_output_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		local_output_path = "/home/MacOS/"+bag_base+".pt"
		print("local_output_path "+ local_output_path)
		output_path = os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt')
		print("output_path ", output_path)
		torch.save(features, local_output_path )
		bucket = storage_client.bucket("oncomerge")
		blob = bucket.blob(output_path)
		blob.upload_from_filename(local_output_path )
		os.remove(slide_file_path)
		file_name =  os.path.splitext(os.path.basename(h5_file_path))[0]
		os.remove( "/home/MacOS/"+ file_name+ '.h5')
		os.remove("/home/MacOS/h5_files/"+os.path.basename(output_path))
		os.remove(local_output_path)



