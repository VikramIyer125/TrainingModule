import numpy as np
import os
import copy
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import torch.optim as optim
import tarfile
import nibabel as nib
import shutil
from skimage.transform import rescale, resize, downscale_local_mean
import ipdb
from sklearn.model_selection import KFold
import pickle
import pdb
from torch.utils.data import random_split
import gc

"""
from bqapi.util import *
from bqapi.comm import BQCommError
from bqapi.comm import BQSession
"""
# from numba import jit, vectorize, int32

from torchvision.datasets.folder import DatasetFolder
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import warnings 

warnings.filterwarnings("ignore")

from train import *

def normalization(planes, norm='gn'):
	if norm == 'bn':
		m = nn.BatchNorm3d(planes)
	elif norm == 'gn':
		m = nn.GroupNorm(4, planes)
	elif norm == 'in':
		m = nn.InstanceNorm3d(planes)
	else:
		raise ValueError('normalization type {} is not supported'.format(norm))
	return m

def run_module(input_path_dict, output_folder_path):
	scan_foler = ''
	GT_folder = ''

	#Get paths for Module 
	current_directory = os.getcwd()

	with open(input_path_dict['Input Images']) as f: 
		lines = f.read()
		first = lines.split('\n', 1)[0]
		scan_folder = first

	with open(input_path_dict['GT seg']) as f: 
		lines = f.read()
		first = lines.split('\n', 1)[0]
		GT_folder = first

	images_path_file = os.path.join(current_directory, scan_folder)
	segs_path_file = os.path.join(current_directory, 'src/'+GT_Folder) 

	model_file = input_path_dict['Model path']


	"""
	bq = BQSession().init_local("user", "pass", bisque_root="https://bisque.ece.ucsb.edu")
	scan_url = bq.service_url('image_service', path=id)
	bq.fetchblob(scan_url, scan_folder+'/')

	gt_url = bq.service_url('image_service', path=id)
	bq.fetchblob(gt_url, GT_folder+'/')
	"""

	#Initialize default values
	torch.backends.cudnn.benchmark = True
	dtype = torch.FloatTensor
	do_yellow = False
	do_prep = False
	use_gpu = False
	N = 256
	N1 = 128
	batch_size = 1
	num_epochs = 1
	start_k = 0
	reload_k_epoch = False
	reload_k = 4
	reload_l = 175
	lr = 0.001
	num_classes = 7
	use_amp = True
	kfold = KFold(shuffle=True)
	crit = 'hard_per_im'
	
	#Initialize Dataset and model 

	nph_dataset = NPHDataset(images_path_file, segs_path_file)

	"""
	try: 
		unet = torch.jit.load(model_file)
		unet.cuda()
	except: 
		unet = Unet()
		unet.cuda()
	"""

	unet = Unet()
	if use_gpu:
		unet.cuda()
	
	if crit == 'hard_per_im':
		criterion = criterions.hard_per_im_cross_entropy
	elif crit == 'hard':
		criterion = criterions.hard_cross_entropy
	elif crit == 'weighted_BCE':
		criterion = criterions.weighted_BCE
	elif crit == 'weighted_hard':
		criterion = criterions.weighted_hard
	
	#Make changes to model 
	net = copy.deepcopy(unet)
	net.convd1.conv1 = ConvD(1,16,0.5,'gn',first=True)
	net.convd1.conv1.weight = nn.Parameter(unet.convd1.conv1.weight[:,1,:,:,:].unsqueeze(1))
	net.seg3 = nn.Conv3d(128, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
	net.seg2 = nn.Conv3d(64, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
	net.seg1 = nn.Conv3d(32, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
	net.seg3.weight.data[0:5] = unet.seg3.weight.data[0:5,:,:,:,:]
	net.seg2.weight.data[0:5] = unet.seg2.weight.data[0:5,:,:,:,:]
	net.seg1.weight.data[0:5] = unet.seg1.weight.data[0:5,:,:,:,:]
	net.seg3.weight.data[5:num_classes] = unet.seg3.weight.data[0:num_classes-5,:,:,:,:]
	net.seg2.weight.data[5:num_classes] = unet.seg2.weight.data[0:num_classes-5,:,:,:,:]
	net.seg1.weight.data[5:num_classes] = unet.seg1.weight.data[0:num_classes-5,:,:,:,:]
	net.seg1.weight = nn.Parameter(net.seg1.weight)
	net.seg2.weight = nn.Parameter(net.seg2.weight)
	net.seg3.weight = nn.Parameter(net.seg3.weight)
	del unet
	if use_gpu: 
		net.to("cuda")
	

	#Initialize Optimizer 
	optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=True, weight_decay=0.0001)
	if use_gpu: 
		scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	#Start training, no K fold 
	train_split = 0.8
	random_seed = 42
	
	indices = list(range(len(nph_dataset)))
	split_point = int(train_split * len(nph_dataset))
	train_indicies, test_indicies = indices[:split_point], indices[split_point:]

	train_subsampler = torch.utils.data.SubsetRandomSampler(train_indicies)
	test_subsampler = torch.utils.data.SubsetRandomSampler(test_indicies)

	trainloader = torch.utils.data.DataLoader(
					  nph_dataset, num_workers=0, pin_memory=True, batch_size=batch_size,
					  persistent_workers=0, prefetch_factor=2, sampler=train_subsampler)
	testloader = torch.utils.data.DataLoader(
					  nph_dataset, num_workers=0, pin_memory=True, batch_size=batch_size, 
					  persistent_workers = 0, sampler=test_subsampler)
	for e in range(num_epochs): 
		print("-----"+str(e+1)+"-------")
		current_loss = 0.0
		for i, data in enumerate(trainloader, 0): 
			_, inputs, labels, weights, scan_name = data
			if use_gpu: 
				with autocast(enabled=use_amp): 
					outputs = net(inputs.to('cuda'))
					if crit.startswith("weighted"): 
						loss = criterion(outputs, labels, weights)
					else: 
						loss = criterion(outputs, labels.cuda())
			else: 
				outputs = net(inputs)
				if crit.startswith("weighted"): 
						loss = criterion(outputs, labels, weights)
				else: 
					loss = criterion(outputs, labels)
			current_loss += loss.item()
			print("Loss: ", current_loss)
			if use_gpu: 
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad(set_to_none=True)
			else: 
				loss.backward()
				optimizer.step()
		print("Epoch " +str(e+1)+" total training loss: " + str(current_loss))
		print("Epoch " +str(e+1)+" average training loss: " + str(current_loss/len(train_indicies)))
	print("Finished training")
	net.eval()
	with torch.no_grad():
		test_loss = 0.0
		for i, test_data in enumerate(testloader, 0): 
			_, inputs, labels, weights, scan_name = test_data
			if use_gpu: 
				outputs = net(inputs.to('cuda'))
			else: 
				outputs = net(inputs)
			if crit.startswith("weighted"): 
				loss = criterion(outputs, labels, weights)
			else: 
				if use_gpu: 
					loss = criterion(outputs, labels.cuda())
				else: 
					loss = criterion(outputs, labels)
			test_loss += loss.item()
			print("Current loss: ", test_loss)
	print("Test loss: " + str(test_loss))
	print("Average test loss: ", test_loss/len(test_indicies))
	output_paths_dict = {}
	output_paths_dict['Model file'] = os.path.join(output_folder_path, "model_final.pt")
	torch.save(net.state_dict(), "model_final.pt")
	return output_paths_dict



if __name__== "__main__": 

	#N17-3.1.1.scan..nii.gz
	#N17-3.Scan1.Subscan1.nii.gz
	input_path_dict = {}
	input_path_dict['Input Images'] = 'images_path.txt'
	input_path_dict["GT seg"] = "GT_seg.txt"
	input_path_dict["Model path"] = "model_final.pt"
	current_dir = os.getcwd()
	output_paths_dict = run_module(input_path_dict, current_dir)