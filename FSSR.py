import os,  time,  copy,  warnings, math

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# remove
import torch.nn.functional as F
import torchvision
import torchvision.io
import torchvision.transforms as transforms
# - remove

import utils
from models import *
from meta import Meta
from finetuner import FineTuner
from datasets.BasicDataset import BasicDataset
from datasets.TaskDataset import TaskDataset
from loss_functions import perceptionLoss, ultimateLoss, VGGPerceptualLoss

warnings.filterwarnings("ignore", message="torch.gels is deprecated in favour of")

# Use GPU if available
print('is cuda available?', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MAMLtrain(model, epochs_nb, trainloader, validloader, batch_size=1, verbose=True):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 6500000.0
    train_size = len(trainloader)
    valid_size = len(validloader)
    print("Training start", flush=True)

    for epoch in range(epochs_nb):
        # Verbose 1
        if verbose:
            print("Epoch [" + str(epoch + 1) + " / " + str(epochs_nb) + "]", flush=True)
            print("-" * 10, flush=True)

        # Training
        running_loss = 0.0
        verbose_loss = 0.0
        for i, data in enumerate(trainloader):
            support_data, support_label, query_data, query_label = data[0].to(device), data[1].to(device), data[2].to(
                device), data[3].to(device) # data [ d.to(device) for d in data]
            loss = model(support_data, support_label, query_data, query_label)
            print('loss:', loss)

            if i % 20 == 0:
                print("Batch " + str(i) + " / " + str(int(train_size)), flush=True)
            running_loss += loss
            verbose_loss += loss
            if i % 100 == 0 and i != 0:
                print("Loss over last 100 batches: " + str(verbose_loss / (100 * batch_size)), flush=True)
                verbose_loss = 0.0

        # Verbose 2
        if verbose:
            epoch_loss = running_loss / (train_size * batch_size)
            print(" ", flush=True)
            print(" ", flush=True)
            print("****************")
            print('Training Loss: {:.7f}'.format(epoch_loss), flush=True)

        # Validation
        running_loss = 0.0
        verbose_loss = 0.0
        for i, data in enumerate(validloader):
            support_data, support_label, query_data, query_label = data[0].to(device).squeeze(0), data[1].to(
                device).squeeze(0), data[2].to(device), data[3].to(device) # same
            loss = model.finetuning(support_data, support_label, query_data, query_label)

            running_loss += loss

        # Verbose 3
        if verbose:
            epoch_loss = running_loss / (valid_size * batch_size)
            print('Validation Loss: {:.7f}'.format(epoch_loss), flush=True)

        # Copy the model if it gets better with validation
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model.state_dict())

    # Verbose 4
    if verbose:
        time_elapsed = time.time() - since
        print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60), flush=True)
        print("Best validation loss: " + str(best_loss), flush=True)

    model.load_state_dict(best_model)  # In place anyway
    return model  # Returning just in case


def save_model_state(state_dict, fp, verbose=True):
    torch.save(state_dict, fp)
    print('Weights saved to: %s' % fp) if verbose else 0


def meta_train(train_fp, valid_fp, load=None, scale=8, shots=10, bs=1, epochs=20,
    lr=0.0001, meta_lr=0.00001, save='out.pth'):

    logger = utils.Logger('yes.log')
    print('Running!', file=logger)

    autoencoder = EDSR(scale=scale)

    meta_learner = Meta(autoencoder.getconfig(), update_lr=lr, meta_lr=meta_lr, update_step=10,
        update_step_test=10).to(device)

    if load:
        weights = torch.load(load)
        meta_learner.load_state_dict(weights)
        print('Weights loaded from %s' % load)

    train_set = TaskDataset(train_fp, shots, scale, augment=True, resize=(256, 512))
    train_dl = DataLoader(train_set, batch_size=bs, num_workers=4, shuffle=True)
    print('Found %i images in training set.' % len(train_set))

    valid_set = TaskDataset(valid_fp, shots, scale, resize=(256, 512))
    valid_dl = DataLoader(valid_set, batch_size=bs, num_workers=2, shuffle=False)
    print('Found %i images in validation set.' % len(valid_set))

    print(autoencoder, flush=True)

    meta_learner = MAMLtrain(meta_learner, epochs, train_dl, valid_dl, bs)
    save_model_state(meta_learner.state_dict(), save)


## Upscale - Using the model
def MAMLupscale(in_path, out_path, weights_path, learning_rate, batch_size, verbose, device_name, benchmark=False, network='EDSR'):
    if device_name == "cuda_if_available":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    if network == 'EDSR':
        model = EDSR()

    config = model.getconfig()

    meta_learner = Meta(config, learning_rate, 0, 10, 10) # Meta learning rate = 0 so no update is performed at test time. Anyway, it doesn't matter which value is given here because it won't be used at test time.
    meta_learner.load_state_dict(torch.load(weights_path))

    del model

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
        if not(os.path.exists(os.path.join(out_path, 'without/'))):
            os.mkdir(os.path.join(out_path, 'without/'))
        if not(os.path.exists(os.path.join(out_path, 'labels/'))):
            os.mkdir(os.path.join(out_path, 'labels/'))
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


def finetuneMaml(train_path, valid_path, batch_size, epoch_nb, learning_rate, meta_learning_rate, load_weights, save_weights, finetune_depth, network='EDSR'):
    if network == 'EDSR':
        model = EDSR().to(device)
    config = model.getconfig()
    # Data loading
    transform = transforms.ToTensor()
    scale_factor = 2
    trainset = utils.DADataset(train_path, transform=transform, num_shot=10, is_valid_file=utils.is_file_not_corrupted,
                               scale_factor=scale_factor, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=4)  # Batch must be composed of images of the same size if >1
    print("Found " + str(len(trainloader) * batch_size) + " images in " + train_path, flush=True)

    validset = utils.FSDataset(valid_path, transform=transform, is_valid_file=utils.is_file_not_corrupted,
                               scale_factor=scale_factor, mode='train')
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Found " + str(len(validloader) * batch_size) + " images in " + valid_path, flush=True)
    meta_learner = Meta(config, learning_rate, meta_learning_rate, 10, 10, load_weights=load_weights).to(device)
    meta_learner = MAMLtrain(meta_learner, epoch_nb, trainloader, validloader, batch_size)
    makeCheckpoint(meta_learner, save_weights)

    # ToDo: Rework Data Augmentation.
    # ToDo: Then, maybe modify meta.py code to be able to only train on the latest layers of the neural network.

    return


def model_train(train_path, valid_paths,                            # data
                load_weights=None, model_name='EDSR', scale=4,      # model
                epochs=10, learning_rate=0.0001, batch_size=16,     # hyper-params
                name='', save_weights='weights.pt', verbose=True):  # run setting
    if not name:
        if load_weights:
            name = '%s_finetuning' % (load_weights.split('/')[-1].split('pt')[0])
        else:
            name = '%sx%i_training' % (model_name, scale)
        name += '-%s-%ie-bs%i' % (train_path.replace('_', '-'), epochs, batch_size)

    logger = utils.Logger('%s.log' % name)
    print('Running [%s]' % name, file=logger)

    if model_name == 'EDSR':
        model = EDSR(scale=scale).to(device)

    perception_loss =  VGGPerceptualLoss().to(device)

    if load_weights is not None:
        model.load_state_dict(torch.load(load_weights))
        print("Loaded weights from: " + str(load_weights))

    def train(model, epochs, train_loader, valid_loaders, optimizer):
        since = time.time()
        best_model = copy.deepcopy(model.state_dict())
        best_loss = math.inf

        for epoch in range(epochs):
            print("Epoch [" + str(epoch+1) + " / " + str(epochs) + "]")

            # Training
            running_loss = 0.0
            for i, data in (t := tqdm(enumerate(train_loader), total=len(train_loader))):
                x, y = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = perception_loss(y_hat, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                t.set_description('loss: %.4f' % loss.item())

            print('Training loss: %.4f' % (running_loss/len(train_loader)), file=logger)

            # Validation
            with torch.no_grad():
                for valid_loader, loader_name in zip(valid_loaders, valid_paths):
                    running_loss = 0.0
                    for i, data in enumerate(valid_loader):
                        x, y = data[0].to(device), data[1].to(device)
                        y_hat = model(x)
                        loss = perception_loss(y_hat, y)
                        running_loss += loss.item()

                    epoch_loss = running_loss/len(valid_loader)
                    print('Validation loss on %s: %.4f' % (loader_name, epoch_loss), file=logger)

            # Copy the model if it gets better with validation
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training finished in %fm %fs' % (time_elapsed//60, time_elapsed%60))
        print('Best validation loss: %.4f' % best_loss)

        return best_model

    resize = (256, 512) # force resize since we are working with batch_size > 1

    train_set = BasicDataset(train_path, training=True, resize=resize, scale_factor=scale)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_loaders = []
    for valid_path in valid_paths:
        valid_set = BasicDataset(valid_path, training=False, resize=resize, scale_factor=scale)
        valid_loaders.append(torch.utils.data.DataLoader(valid_set,
            batch_size=batch_size, shuffle=True, num_workers=2))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    best_model_state_dict = train(model, epochs, train_loader, valid_loaders, optimizer)
    save_model_state(best_model_state_dict, save_weights)


def upscale(load_weights, input, out):
    edsr = EDSR().to(device)
    print(edsr)

    if load_weights is not None:
        edsr.load_state_dict(torch.load(load_weights))
        print("Loaded weights from: " + str(load_weights), flush=True)

    label_path = os.path.join(out, 'labels/')
    if not(os.path.exists(label_path)):
        os.mkdir(label_path)

    constructed_path = os.path.join(out, 'constructed/')
    if not(os.path.exists(constructed_path)):
        os.mkdir(constructed_path)

    validset = BasicDataset(input, training=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False, num_workers=2)

    image_count = len(validloader)
    with torch.no_grad():
        for i, data in enumerate(validloader):
            query, label = data[0].to(device), data[1].to(device)
            output = edsr(query)

            # name = validloader.get_image_name(i)
            print('upscaling [', round(i/image_count*100, 2), '%]', i)

            output = transforms.ToPILImage(mode='RGB')(output.squeeze(0).cpu())
            output.save(os.path.join(constructed_path, str(i) + '.png'))

            label = transforms.ToPILImage(mode='RGB')(label.squeeze(0).cpu())
            label.save(os.path.join(label_path, str(i) + '.png'))
    return


