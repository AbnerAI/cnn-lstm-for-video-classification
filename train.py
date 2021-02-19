import os
import torch
import random
import argparse
import numpy as np
import tensorboardX
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from utils import AverageMeter, calculate_accuracy

def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device, batch_size):
    model.train()
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()

    # iter_source = iter(data_loader)
    num_iter  = len(data_loader)
    for batch_idx, sample in enumerate(data_loader):
        if batch_idx==12:
            break
        data = sample['imgs'].view(-1, batch_size, 3, 224, 224)
        targets = sample["label"] 
		# path_ = sample["path"]
		# print(imgs.shape)
		# print(label)
    # for batch_idx in range(1, num_iter):
        # data, targets = iter_source.next()
        # data, targets = imgs.to(device), label.to(device)
        outputs = model(data)
        # print(outputs.shape)
        # print(outputs)
        # exit(0)
        loss = criterion(outputs, targets) # outputs.shape: [1, 2] & targets.shape: [1]
        acc = calculate_accuracy(outputs, targets)
        
        train_loss += loss.item()
        # print('loss item: ', loss.item())
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
            train_loss = 0.0
    
    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg  