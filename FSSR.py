import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
import copy
from PIL import Image, ImageFile
from datetime import datetime
import sys
import cv2
import torchvision.io

os.chdir(os.path.dirname(os.path.realpath(__file__)))
from models import *
from meta import Meta
import utils
from loss_functions import perceptionLoss, ultimateLoss
#from benchmark.PSNR import meanPSNR
#from benchmark.SSIM import meanSSIM

## Training
def train(train_path, valid_path, batch_size, epoch_nb, learning_rate, meta_learning_rate, save_path, verbose, weights_load=None, loss_func='MSE', loss_network='vgg16', network='EDSR', num_shot=10):

    ## Main loop
    def MAMLtrain(model, loss_function, epochs_nb, num_shot=10):
        since = time.time()
        best_model = copy.deepcopy(model.state_dict())
        best_loss = 6500000.0
        train_size = len(trainloader)
        valid_size = len(validloader)
        print("Training start", flush=True)

        for epoch in range(epochs_nb):
            # Verbose 1
            if verbose:
                print("Epoch [" + str(epoch+1) + " / " + str(epochs_nb) + "]", flush=True)
                print("-" * 10, flush=True)

            # Training
            running_loss = 0.0
            verbose_loss = 0.0
            for i, data in enumerate(trainloader):
                support_data, support_label, query_data, query_label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                #print(query_label.size())
                loss = meta_learner(support_data, support_label, query_data, query_label)
                print(loss)

                if i%20 == 0:
                    print("Batch " + str(i) + " / " + str(int(train_size)), flush=True)
                running_loss += loss
                verbose_loss += loss
                if i% 100 == 0 and i !=0:
                    print("Loss over last 100 batches: " + str(verbose_loss/(100*batch_size)), flush=True)
                    verbose_loss = 0.0

            # Verbose 2
            if verbose:
                epoch_loss = running_loss / (train_size*batch_size)
                print(" ", flush=True)
                print(" ", flush=True)
                print("****************")
                print('Training Loss: {:.7f}'.format(epoch_loss), flush=True)

            # Validation
            running_loss = 0.0
            verbose_loss = 0.0
            for i, data in enumerate(validloader):
                support_data, support_label, query_data, query_label = data[0].to(device).squeeze(0), data[1].to(device).squeeze(0), data[2].to(device).squeeze(0), data[3].to(device).squeeze(0)

                loss = model.finetuning(support_data, support_label, query_data, query_label)

                running_loss += loss

            # Verbose 3
            if verbose:
                epoch_loss = running_loss / (valid_size*batch_size)
                print('Validation Loss: {:.7f}'.format(epoch_loss), flush=True)

            # Copy the model if it gets better with validation
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
        # Verbose 4
        if verbose:
            time_elapsed = time.time() - since
            print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60), flush=True)
            print("Best validation loss: " + str(best_loss), flush=True)

        model.load_state_dict(best_model) # In place anyway
        return model # Returning just in case

    def makeCheckpoint(model, save_path): # Function to save weights
        torch.save(model.state_dict(), save_path)
        if verbose:
            print("Weights saved to: " + save_path, flush=True)
        return

    ## Init training
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scale_factor = 2

    # Setup model and hyper parameters
    if network == 'EDSR':
        autoencoder = EDSR(scale=scale_factor)

    config = autoencoder.getconfig()

    meta_learner = Meta(config, learning_rate, meta_learning_rate, 10, 10).to(device)  #ToDo

    transform = torchvision.transforms.Compose([transforms.ToTensor()])

    # Data loading
    trainset = utils.DADataset(train_path, transform=transform, num_shot=10, is_valid_file=utils.is_file_not_corrupted, scale_factor=scale_factor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4) # Batch must be composed of images of the same size if >1
    print("Found " + str(len(trainloader)*batch_size) + " images in " + train_path, flush=True)

    validset = utils.DADataset(valid_path, transform=transform, num_shot=10, is_valid_file=utils.is_file_not_corrupted, scale_factor=scale_factor)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Found " + str(len(validloader)*batch_size) + " images in " + valid_path, flush=True)

    # ToDO: Change weights loading.
    if weights_load is not None: # Load weights for further training if a path was given.
        autoencoder.load_state_dict(torch.load(weights_load))
        print("Loaded weights from: " + str(weights_load), flush=True)
    autoencoder.to(device)

    print(autoencoder, flush=True)

    if loss_func == "MSE": # Get the appropriate loss function.
        loss_function = nn.MSELoss()
    elif loss_func == "perception":
        loss_function = perceptionLoss(pretrained_model=loss_network)
    elif loss_func == "ultimate":
        loss_function = ultimateLoss(pretrained_model=loss_network)

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, amsgrad=True)

    # Start training
    MAMLtrain(meta_learner, loss_function, epoch_nb, num_shot=num_shot)
    makeCheckpoint(autoencoder, save_path)
    return

