import os
from argparse import ArgumentParser

from FSSR import meta_train, finetuneMaml, MAMLupscale, model_train, upscale
from evaluation import evaluation
from utils import require_args, summarize_args

os.chdir(os.path.dirname(os.path.realpath(__file__)))

## Parser
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-m', '--mode', choices=['meta_train', 'meta_upscale', 'finetune_maml', 'evaluation', 'upscale_video', 'model_train', 'upscale'], required=True,
        help="The name of the mode to run") # make positional arg ; replace '_' with '-' ; rename 'meta_train' to 'vanilla_train'

    parser.add_argument('--name',
        help='Name of the operation that is run, used for naming .log and .pt files')

    parser.add_argument('--device', default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda_if_available'],
        help='Device identifier to run the process on.') # remove

    parser.add_argument('--model',  default='EDSR', choices=['EDSR'],
        help='Indicates which neural network to use.') # remove
    parser.add_argument('--batch-size', type=int,
        help='The number of images for each training iteration as an integer.')
    parser.add_argument('--scale', type=int,
        help='The scaling factor for an upscaling task.')
    parser.add_argument('--epochs', type=int,
        help='Number of epochs for training i.e number of times the training set is iterated over.')

    parser.add_argument('--train-folder',
        help='Path to the folder containing the images of the training set.')
    parser.add_argument('--valid-folders', nargs='+', type=str,
        help='Path to the folder containing the images of the validation set. Can be multiple paths.')
    parser.add_argument('--load-weights', default=None,
        help='Path to the weights to continue training, perform upscaling or evaluate performance.')
    parser.add_argument('--save-weights', default='./weights/test.pt',
        help='Path to save the weights after training (.pt).')

    # ~todo
    parser.add_argument('--input', default='../CelebA/', # '../../../../mnt/data/prim_project/dataset/FSSR/CelebA/'
        help="Path to the directory containing the images to upscale. Only used for upscaling or evaluation.")
    parser.add_argument('--output', default='./out/',
        help="Destination directory for the benchmark in 'evaluation' mode or upscaled images in 'upscale' mode.")

    parser.add_argument('--verbose', default=True, type=bool, choices=[True, False],
        help="Whether the script print info in stdout.")


    parser.add_argument('--learning_rate', default=0.0001, type=float,
        help="Learning rate for training with Adam optimizer. Only for 'meta_train' & 'meta_upscale' mode.")
    parser.add_argument('--meta_learning_rate', default=0.00001,
        help="Learning rate of the meta training.")
    parser.add_argument('--loss', default='MSE', choices=['MSE', 'perception', 'ultimate'],
        help="The loss function to use for training. Percepion loss uses a loss network that can be chosen with --loss_network arg. Only for 'train' mode.")
    parser.add_argument('--loss_network', default='vgg16', choices=['vgg16', 'vgg19', 'resnet18'],
        help="The loss network used for perceptual loss computing. Only for 'train' mode")
    parser.add_argument('--finetune_depth', default=0, type=int,
        help="Number of parameter tensors to be finetuned in finetune_maml mode. 0 to modify all layers.")

    opt = parser.parse_args()
    require = require_args(opt)
    summarize = summarize_args(opt, {
        'train_folder': lambda x: 'training set: %s' % x,
        'valid_folders': lambda x: 'validation sets: %s' % ' | '.join(x),
        'epochs': lambda x: 'epochs nb: %i' % x,
        'scale': lambda x: 'scale factor: x%i' % x,
        'batch_size': lambda x: 'batch size: %i' % x,
        'load_weights': lambda x: 'loading weights?: %s %s' % (bool(x), x if x else '' ),
    })

    if opt.mode == 'meta_train':
        require('train_folder', 'valid_folders', 'epochs', 'batch_size', 'scale')
        summarize('train_folder', 'valid_folders', 'epochs', 'scale', 'batch_size', 'load_weights')
        meta_train(train_fp=opt.train_folder,
                   valid_fp=opt.valid_folders[0],
                   load=opt.load_weights,
                   scale=opt.scale,
                   bs=opt.batch_size,
                   epochs=opt.epochs,
                   lr=opt.learning_rate,
                   meta_lr=opt.meta_learning_rate,
                   save=opt.save_weights)

    elif opt.mode == 'finetune_maml':
        finetuneMaml(train_path=opt.train_folder,
                     valid_path=opt.valid_folder,
                     batch_size=opt.batch_size,
                     epochs=opt.epochs,
                     learning_rate=opt.learning_rate,
                     meta_learning_rate=opt.meta_learning_rate,
                     load_weights=opt.load_weights,
                     save_weights=opt.save_weights,
                     finetune_depth=opt.finetune_depth,
                     network=opt.model)

    elif opt.mode == 'meta_upscale':
        MAMLupscale(in_path=opt.input,
                    out_path=opt.output,
                    weights_path=opt.load_weights,
                    learning_rate=opt.learning_rate,
                    batch_size=opt.batch_size,
                    verbose=opt.verbose,
                    device_name=opt.device,
                    network=opt.model)

    elif opt.mode == 'evaluation':
        evaluation(in_path=opt.input, out_path=opt.output, verbose=True)

    elif opt.mode == 'upscale_video':
        pass

    elif opt.mode == 'model_train':
        require('train_folder', 'valid_folders', 'epochs', 'batch_size', 'scale')
        summarize('train_folder', 'valid_folders', 'epochs', 'scale', 'batch_size', 'load_weights')
        model_train(train_path=opt.train_folder,
                    valid_paths=opt.valid_folders,
                    epochs=opt.epochs,
                    batch_size=opt.batch_size,
                    load_weights=opt.load_weights,
                    save_weights=opt.save_weights,
                    name=opt.name,
                    scale=opt.scale)

    elif opt.mode == 'upscale':
        upscale(load_weights=opt.load_weights, input=opt.input, out=opt.output)
