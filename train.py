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
        data = sample['imgs'].permute([1,0,2,3,4])#.view(-1, batch_size, 3, 224, 224)
        targets = sample["label"] 
        outputs = model(data)
   
        loss = criterion(outputs, targets) # outputs.shape: [1, 2] & targets.shape: [1]
        acc = calculate_accuracy(outputs, targets)
        
        train_loss += loss.item()
        losses.update(loss.item(), data.size(1)) # data.size(1) is batch-size 
        accuracies.update(acc, data.size(1))
        
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