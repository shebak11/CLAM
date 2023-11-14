from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange
from google.cloud import storage
from tiffslide import TiffSlide

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path
		

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		gs_slide_file_path,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.file_path = file_path
		self.pretrained=pretrained
		#self.wsi = wsi
		self.gs_slide_file_path=gs_slide_file_path
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms  
		
		file_name =  os.path.splitext(os.path.basename(file_path))[0]
		print("h5 file " + file_name)
		#local_file_path = "/home/MacOS/"+ file_name+ '.h5'
		#local_file_path = "/home/MacOS/"+ "TCGA-3L-AA1B-01A-01-TS1.9C415218-D5B4-4945-B243-F42A4C8C0484"+ '.h5'
		#local_file_path = "/home/MacOS/TCGA-3L-AA1B-01A-01-TS1.9C415218-D5B4-4945-B243-F42A4C8C0484.h5"
		#self.file_path = local_file_path

		storage_client = storage.Client()
		bucket = storage_client.bucket("oncomerge")
		gs_path = file_path

		#blob = bucket.blob(gs_path)
		#blob.download_to_filename(self.file_path )
        
        
		blob = bucket.blob(self.file_path)
		with blob.open("rb") as f:
			with h5py.File(f,'r') as hdf5_file:
			 dset = hdf5_file['coords']    
			 #self.dset = f['coords'][:]  
			 self.coord=dset[0] 
			 self.patch_level = hdf5_file['coords'].attrs['patch_level']
			 self.patch_size = hdf5_file['coords'].attrs['patch_size']
			 self.length = len(dset)
			 if target_patch_size > 0:
			     self.target_patch_size = (target_patch_size, ) * 2
			 elif custom_downsample > 1:
			     self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			 else:
			     self.target_patch_size = None   
		"""
		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']    
			#self.dset = f['coords'][:]  
			self.coord=dset[0] 
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None    
		"""
		self.summary()
		#self.coord=self.dset[0]     
		#coord=self.dset[0]   
		
				
		
		#self.length=512
		#self.img=self.wsi.read_region(self.coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		#self.img = self.wsi.read_region((300, 400), level = 0, size = (512, 512)).convert('RGB')
			
	def __len__(self):
		return self.length

	def summary(self):
        storage_client = storage.Client()
		bucket = storage_client.bucket("oncomerge")
		blob = bucket.blob(self.file_path)
		with blob.open("rb") as f:
		  hdf5_file = h5py.File(f, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		blob = bucket.blob(self.file_path)
		with blob.open("rb") as f:
		  with h5py.File(f,'r') as hdf5_file:
		      coord = hdf5_file['coords'][idx]
		#hdf5_file = h5py.File(self.file_path, "r")
		#coord=self.coord
		#coord=hdf5_file['coords'][idx]
		#region = slide.read_region((300, 400), 0, (512, 512))
		#img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		#img = self.img
		#img = self.wsi.read_region(location = (coord[0], coord[1]), level = self.patch_level, size = (self.patch_size, self.patch_size)).convert('RGB')
		storage_client = storage.Client()
		bucket = storage_client.bucket("oncomerge")
		blob = bucket.blob(self.gs_slide_file_path)
		with blob.open("rb") as f:
		  wsi = TiffSlide(f)
		  img = wsi.read_region(location = (300, 400), level = self.patch_level, size = (self.patch_size, self.patch_size)).convert('RGB')

		#img = self.wsi.read_region((300, 400), level = 0, size = (512, 512)).convert('RGB')

		#img = self.img
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		#img = torch.tensor(img)
		#coord=[300 , 400]
		#return img, coord
		return img, coord
		#return 5,5

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		#self.df = pd.read_csv(csv_path)
		self.df = pd.read_csv("gs://oncomerge/"+csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		print(list(self.df))
		return self.df['slide_id'][idx]



