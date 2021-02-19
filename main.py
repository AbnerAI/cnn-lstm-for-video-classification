import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import os
import random
import numpy as np

from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
# from dataset import get_training_set, get_validation_set
from dataset import Ultra_Dataset_Train, Ultra_Dataset_Eval
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
	MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose

from torchvision import datasets, transforms
import torch

def load_training(root_path, dir, batch_size, kwargs):
    # transform = transforms.Compose(
    #     [transforms.Resize([256, 256]),
    #      transforms.RandomCrop(224),
    #      transforms.RandomHorizontalFlip(),
    #      transforms.ToTensor()])

	transform = transforms.Compose([transforms.Resize([224, 224]),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
	data = datasets.ImageFolder(root=root_path + dir, transform=transform)

	# if shuffle==True: random 
	train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
	return train_loader

def resume_model(opt, model, optimizer):
	""" Resume model 
	"""
	checkpoint = torch.load(opt.resume_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Model Restored from Epoch {}".format(checkpoint['epoch']))
	start_epoch = checkpoint['epoch'] + 1
	return start_epoch


def get_loaders(opt):
	""" Make dataloaders for train and validation sets
	"""
	# train loader
	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset) 
	if opt.no_mean_norm and not opt.std_norm:
		norm_method = Normalize([0, 0, 0], [1, 1, 1])
	elif not opt.std_norm:
		norm_method = Normalize(opt.mean, [1, 1, 1])
	else:
		norm_method = Normalize(opt.mean, opt.std)
	spatial_transform = Compose([
		# crop_method,
		Scale((opt.sample_size, opt.sample_size)),
		# RandomHorizontalFlip(),
		ToTensor(opt.norm_value), norm_method
	])
	temporal_transform = TemporalRandomCrop(16)
	target_transform = ClassLabel()
	training_data = get_training_set(opt, spatial_transform,
									 temporal_transform, target_transform)
	train_loader = torch.utils.data.DataLoader(
		training_data,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.num_workers,
		pin_memory=True)
	
	# validation loader
	spatial_transform = Compose([
		Scale((opt.sample_size, opt.sample_size)),
		# CenterCrop(opt.sample_size),
		ToTensor(opt.norm_value), norm_method
	])
	target_transform = ClassLabel()
	temporal_transform = LoopPadding(16)
	validation_data = get_validation_set(
		opt, spatial_transform, temporal_transform, target_transform)
	val_loader = torch.utils.data.DataLoader(
		validation_data,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		pin_memory=True)
	return train_loader, val_loader

# /home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H/2
def main_worker():
	transform = transforms.Compose([transforms.Resize([224, 224]),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
	seq_len = 3
	batch_size = 2
	train_set = Ultra_Dataset_Train(data_dir='./lists.txt', transform=transform, seq_len=seq_len)
	train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=0, pin_memory=True)
	
	opt = parse_opts()
	print(opt)
	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# CUDA for PyTorch
	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
	
	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

	# defining model: model = cnnlstm.CNNLSTM(num_classes=opt.n_classes), opt just opt.num_classes.  
	model =  generate_model(opt, device)
	# optimizer
	crnn_params = list(model.parameters())
	optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)
	# optimizer  = optim.Adam(model.parameters(), lr=0.00001)

	criterion = nn.CrossEntropyLoss()

	# resume model
	if opt.resume_path:
		start_epoch = resume_model(opt, model, optimizer)
	else:
		start_epoch = 1
	
	# start training
	for epoch in range(start_epoch, opt.n_epochs + 1):
		print('epoch: ', epoch)
		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, opt.log_interval, device, batch_size)
		continue
		# val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
		
		# saving weights to checkpoint
		if (epoch) % opt.save_interval == 0:
			# scheduler.step(val_loss)
			# write summary
			summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
			summary_writer.add_scalar(
				'losses/val_loss', val_loss, global_step=epoch)
			summary_writer.add_scalar(
				'acc/train_acc', train_acc * 100, global_step=epoch)
			summary_writer.add_scalar(
				'acc/val_acc', val_acc * 100, global_step=epoch)

			state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
			print("Epoch {} model saved!\n".format(epoch))


if __name__ == "__main__":
	main_worker()