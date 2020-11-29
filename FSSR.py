import os,  time,  warnings, math
from statistics import mean
from datetime import timedelta

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Logger, construct_name, load_state, clone_state, save_state
from models import EDSR
from meta import Meta
from finetuner import FineTuner
from datasets import BasicDataset, TaskDataset
from loss_functions import VGGPerceptualLoss

warnings.filterwarnings("ignore", message="torch.gels is deprecated in favour of")

print('Is cuda available? %s' % torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def meta_train_loop(model, epochs, train_dl, valid_dl, logs):
    since = time.time()
    best_model = clone_state(model)
    best_loss = math.inf

    for epoch in range(epochs):
        print('Epoch [%i/%i]' % (epoch + 1, epochs))

        losses = []
        for data in (t := tqdm(train_dl)):
            x_spt, y_spt, x_qry, y_qry = [d.to(device) for d in data]
            loss = model(x_spt, y_spt, x_qry, y_qry)
            losses.append(loss)
            t.set_description('Train loss: %.5f mean(%.5f)' % (loss, mean(losses)))
        print('Training loss: %.5f' % mean(losses), file=logs)

        valid_losses = []
        # It's safe without no_grad() since MAML takes care of cloning our model.
        for data in (t := tqdm(valid_dl)):
            x_spt, y_spt, x_qry, y_qry = [d.to(device) for d in data]
            loss = model.finetuning(x_spt.squeeze(0), y_spt.squeeze(0), x_qry, y_qry)
            valid_losses.append(loss)
            t.set_description('Validation loss: %.5f mean(%.5f)' % (loss, mean(valid_losses)))
        print('Validation loss: %.5f' % mean(valid_losses), file=logs)

        valid_loss = mean(valid_losses)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = clone_state(model)

    time_elapsed = time.time() - since
    print('Training finished in %s' % timedelta(seconds=int(time_elapsed)))
    print('Best validation loss: %.4f' % best_loss)

    return best_model


def meta_train(train_fp, valid_fp, load=None, scale=8, shots=10, bs=1, epochs=20, lr=0.0001,
    meta_lr=0.00001):
    name = construct_name(name='EDSRx%i' % scale, load=load, dataset=train_fp, epochs=epochs,
        bs=bs, action='meta')
    logs = Logger(name + '.logs')
    print('Running <%s>' % name, file=logs)

    autoencoder = EDSR(scale=scale).getconfig()
    meta_learner = Meta(autoencoder, update_lr=lr, meta_lr=meta_lr, update_step=10,
        update_step_test=10).to(device)

    if load:
        load_state(meta_learner, load)
        print('Weights loaded from %s' % load, file=logs)

    train_set = TaskDataset(train_fp, shots, scale, augment=True, style=True, resize=(256, 512))
    train_dl = DataLoader(train_set, batch_size=bs, num_workers=4, shuffle=True)
    print('Found %i images in training set.' % len(train_set))

    valid_set = TaskDataset(valid_fp, shots, scale, augment=False, style=True, resize=(256, 512))
    valid_dl = DataLoader(valid_set, batch_size=1, num_workers=2, shuffle=False)
    print('Found %i images in validation set.' % len(valid_set))

    meta_learner = meta_train_loop(meta_learner, epochs, train_dl, valid_dl, logs)
    save_state(meta_learner, name + '.pth')
    print('Saved model to %s.pth' % name, file=logs)


def models_test(test_fp, model_fps, scale, shots, lr=0.0001, epochs=10):
    name = construct_name(load='%s<eval>' % 'vs'.join(model_fps), dataset=test_fps,
        epochs=epochs, bs=shots, action='test')
    logs = Logger(name + '.logs')
    print('Testing <%s> on %s' % ('> <'.join(model_fps), test_fp), file=logs)

    config = EDSR(scale=scale).getconfig()
    models = []
    for model_fp in model_fps:
        if 'meta' in model_fp:
            models.append(Meta(config, update_lr=lr, meta_lr=0, update_step=0,
                update_step_test=epochs).to(device))
            load_state(model, model_fp)
        else:
            models.append(Meta(config, update_lr=lr, meta_lr=0, update_step=0,
                update_step_test=epochs, load_weights=model_fp).to(device))

    test_set = BasicDataset(test_fp, scale, augment=False, style=False, resize=(256, 512))
    test_dl = DataLoader(test_set, batch_size=shots, num_workers=4, shuffle=True)
    print('Found %i images in test set.' % len(test_set))

    losses = {name: [] for name in model_fps}
    for data in tqdm(test_dl):
        x, y = [d.to(device) for d in data]
        y_spt, y_qry = y[:-1], y[-1]
        x_spt, x_qry = x[:-1], x[-1]

        for model, name in zip(models, model_fps):
            loss = model.finetuning(x_spt, y_spt, x_qry, y_qry)
            losses[name].append(loss)
    print(losses, file=logs)



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


def vanilla_train_loop(model, loss_function, optimizer, epochs, train_dl, valid_dls, valid_fps, logs):
    since = time.time()
    best_model = clone_state(model)
    best_loss = math.inf

    for epoch in range(epochs):
        print('Epoch [%i/%i]' % (epoch + 1, epochs))

        losses = []
        for data in (t := tqdm(train_dl)):
            x, y = [d.to(device) for d in data]
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_function(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            t.set_description('Train loss: %.5f mean(%.5f)' % (loss, mean(losses)))
        print('Training loss: %.5f' % mean(losses), file=logs)

        with torch.no_grad():
            # We compute the validation loss for each validation sets.
            for valid_dl, dl_name in zip(valid_dls, valid_fps):
                valid_losses = []
                for data in valid_dl:
                    x, y = data[0].to(device), data[1].to(device)
                    y_hat = model(x)
                    loss = loss_function(y_hat, y)
                    valid_losses.append(loss.item())
            valid_loss = mean(valid_losses)
            print('Validation loss on %s: %.4f' % (dl_name, valid_loss), file=logs)

        # Will keep the best model based on the validation loss *on the last* validation set.
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = clone_state(model)

    time_elapsed = time.time() - since
    print('Training finished in %s' % timedelta(seconds=int(time_elapsed)))
    print('Best validation loss: %.4f' % best_loss)

    return best_model


def vanilla_train(train_fp, valid_fps, load=None, scale=8, bs=16, epochs=20, lr=0.0001):
    name = construct_name(name='EDSRx%i' % scale, load=load, dataset=train_fp, epochs=epochs,
        bs=bs, action='vanilla')
    logs = Logger(name + '.logs')
    print('Running <%s> !' % name, file=logs)

    model = EDSR(scale=scale).to(device)
    if load:
        load_state(model, load)
        print('Weights loaded from %s' % load, file=logs)

    loss_function =  VGGPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    train_set = BasicDataset(train_fp, scale, augment=True, style=True, resize=(256, 512))
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

    valid_dls = []
    for fp in valid_fps:
        valid_set = BasicDataset(fp, scale, augment=False, style=True, resize=(256, 512))
        valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=True, num_workers=2)
        valid_dls.append(valid_dl)

    model = vanilla_train_loop(model.to(device), loss_function, optimizer, epochs, train_dl,
        valid_dls, valid_fps, logs)
    save_state(model, name + '.pth')
    print('Saved model to %s.pth' % name, file=logs)


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
