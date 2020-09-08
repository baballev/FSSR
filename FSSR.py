import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import time
import copy
import torchvision.io
import warnings

#os.chdir(os.path.dirname(os.path.realpath(__file__)))
from models import *
from meta import Meta
import utils
from loss_functions import perceptionLoss, ultimateLoss
from finetuner import FineTuner
warnings.filterwarnings("ignore", message="torch.gels is deprecated in favour of")

## Training
def meta_train(train_path, valid_path, batch_size, epoch_nb, learning_rate, meta_learning_rate, save_path, verbose, weights_load=None, loss_func='MSE', loss_network='vgg16', network='EDSR', num_shot=10):

    ## Main loop
    def MAMLtrain(model, epochs_nb, trainloader, validloader):
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
                loss = model(support_data, support_label, query_data, query_label)
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
                support_data, support_label, query_data, query_label = data[0].to(device).squeeze(0), data[1].to(device).squeeze(0), data[2].to(device), data[3].to(device)
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

    meta_learner = Meta(config, learning_rate, meta_learning_rate, 10, 10).to(device)

    transform = torchvision.transforms.Compose([transforms.ToTensor()])

    # Data loading
    trainset = utils.DADataset(train_path, transform=transform, num_shot=10, is_valid_file=utils.is_file_not_corrupted, scale_factor=scale_factor, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4) # Batch must be composed of images of the same size if >1
    print("Found " + str(len(trainloader)*batch_size) + " images in " + train_path, flush=True)

    validset = utils.FSDataset(valid_path, transform=transform, is_valid_file=utils.is_file_not_corrupted, scale_factor=scale_factor, mode='train')
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Found " + str(len(validloader)*batch_size) + " images in " + valid_path, flush=True)

    if weights_load is not None: # Load weights for further training if a path was given.
        meta_learner.load_state_dict(torch.load(weights_load))
        print("Loaded weights from: " + str(weights_load), flush=True)

    print(autoencoder, flush=True)

    del autoencoder

    if loss_func == "MSE": # Get the appropriate loss function.
        loss_function = nn.MSELoss()
    elif loss_func == "perception":
        loss_function = perceptionLoss(pretrained_model=loss_network) # ToDo: Embed this code in the meta learner constructor so as to be able to choose which loss function is used by MAML
    elif loss_func == "ultimate":
        loss_function = ultimateLoss(pretrained_model=loss_network)

    # Start training
    meta_learner = MAMLtrain(meta_learner, epoch_nb, trainloader, validloader)
    makeCheckpoint(meta_learner, save_path)
    return

## Upscale - Using the model
def MAMLupscale(in_path, out_path, weights_path, learning_rate, batch_size, verbose, device_name, benchmark=False, network='EDSR'):
    if device_name == "cuda_if_available":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    if network == 'EDSR':
        autoencoder = EDSR()

    config = autoencoder.getconfig()

    meta_learner = Meta(config, learning_rate, 0, 10, 10) # Meta learning rate = 0 so no update is performed at test time. Anyway, it doesn't matter which value is given here because it won't be used at test time.
    meta_learner.load_state_dict(torch.load(weights_path))
    #meta_learner.eval() # useful uhu?

    del autoencoder

    scale_factor = 2
    transform = transforms.ToTensor()
    testset = utils.FSDataset(in_path, transform=transform, is_valid_file=utils.is_file_not_corrupted, scale_factor=scale_factor, mode='train')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    n = len(testloader)
    if verbose:
        print("Found " + str(n) + " images in " + in_path, flush=True)
        print("Beginning upscaling...", flush=True)
        print("Clock started ", flush=True)

    since = time.time()
    for i, data in enumerate(testloader):

        support_data, support_label, query_data, query_label = data[0].to(device), data[1].to(device), data[2].to(
            device), data[3].to(device)
        query_label = query_label.squeeze(0)
        support_data, support_label = support_data.squeeze(0), support_label.squeeze(0)

        if benchmark:
            pass # ToDo: divide by 2 and then re-upscale
        else:
            pass
        reconstructed, reconstructed_without = meta_learner.test(support_data, support_label, query_data)
        img = transforms.Compose([torchvision.transforms.ToPILImage(mode='RGB')])(reconstructed[0].cpu())
        img_l = transforms.ToPILImage(mode='RGB')(query_label.cpu())
        img_without = transforms.ToPILImage(mode='RGB')(reconstructed_without[0].cpu())
        img_without.save(os.path.join(os.path.join(out_path, 'without/'), str(i) + '.png'))
        img_l.save(os.path.join(os.path.join(out_path, 'labels/'), str(i) + '.png'))
        img.save(os.path.join(out_path, str(i) + ".png"))
        if verbose:
            if i % 100 == 0:
                print("Image " + str(i) + " / " + str(n), flush=True)

    time_elapsed = time.time() - since
    if verbose:
        print("Processed " + str(n) + " images in " + "{:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60), flush=True)
        print("Overall speed: " + str(n/time_elapsed) + " images/s", flush=True)
        print("Upscaling: Done, files saved to " + out_path, flush=True)

    return time_elapsed, n/time_elapsed, n, scale_factor


def finetuneTrain():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edsr = EDSR(2, 16, 2).to(device)
    m = FineTuner(edsr, 6).to(device)
    print(m)

def model_train(train_path, valid_path, epoch_nb=1, batch_size=1, load_weights=None, save_weights='weights.pt', model_name='EDSR'):
    device = torch.device('cuda:0')
    verbose = True
    if model_name == 'EDSR':
        super_res_model = EDSR().to(device)

    if load_weights is not None:
        super_res_model.load_state_dict(torch.load(load_weights))
        print("Loaded weights from: " + str(load_weights), flush=True)

    def train(model, epochs_nb, trainloader, validloader, optimizer):
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
                query, label = data[2].to(device), data[3].to(device)
                optimizer.zero_grad()
                out = model(query)
                loss = F.mse_loss(out, label)
                loss.backward()
                optimizer.step()
                print(loss)

                if i%100 == 0:
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
                query, label =  data[2].to(device), data[3].to(device)

                out = model(query)
                loss = F.mse_loss(out, label)

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

    trainset = utils.DADataset(train_path, num_shot=1, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validset = utils.FSDataset(valid_path, transform=transforms.ToTensor())
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = optim.Adam(super_res_model.parameters(), lr=0.0001, amsgrad=True)
    super_res_model = train(super_res_model, epoch_nb, trainloader, validloader, optimizer)
    makeCheckpoint(super_res_model, save_weights)

    return
