from argparse import ArgumentParser
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
from FSSR import meta_train, finetuner_train, MAMLupscale

## Parser
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['meta_train', 'meta_upscale', 'evaluation', 'upscale_video'])
    parser.add_argument('--device', default='cuda_if_available', choices=['cpu', 'cuda', 'cuda_if_available'], help="Leave default to use the GPU if it is available. CPU can't be used for training without changing the code.")
    parser.add_argument('--input', default='../dataset/FSSR/DIV2K/DIV2K_valid_HR/', help="Path to the directory containing the images to upscale. Only used for upscaling or evaluation.")
    parser.add_argument('--output', default='./out/', help="Destination directory for the benchmark in 'evaluation' mode or upscaled images in 'upscale' mode.")
    parser.add_argument('--verbose', default=True, type=bool, choices=[True, False], help="Wether the script print info in stdout.")
    parser.add_argument('--tmp_folder', default='./benchmark/tmp/', help="Temporary folder to store upscaled images when evaluating a model. Only for 'evaluation' mode.")
    parser.add_argument('--network_name',  default='EDSR', choices=['EDSR'], help="Indicates which network is being used.")
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size i.e the number of images for each training iteration as an integer.")

    parser.add_argument('--load_weights', default=None, help="Path to the weights to continue training, perform upscaling on a set of images or evaluate performance.")
    parser.add_argument('--save_weights', default='./weights/test.pt', help="Path to save the weights after training (.pth). Only for 'train' mode.")
    parser.add_argument('--train_folder', default='../dataset/FSSR/DIV2K/DIV2K_train_HR/', help="Path to the folder containing the images of the training set. Only for 'train' mode.")
    parser.add_argument('--valid_folder', default='../dataset/FSSR/DIV2K/DIV2K_valid_HR/', help="Path to the folder containing the images of the validation set. Only for 'train' mode.")
    parser.add_argument('--epoch_nb', default=10, type=int, help="Number of epochs for training i.e the number of times the whole training set is iterated over as an integer. Only for 'train' mode.")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate for training with Adam optimizer. Only for 'meta_train' & 'meta_upscale' mode.")
    parser.add_argument('--meta_learning_rate', default=0.0001, help="Learning rate of the meta training.")
    parser.add_argument('--loss', default='MSE', choices=['MSE', 'perception', 'ultimate'], help="The loss function to use for training. Percepion loss uses a loss network that can be chosen with --loss_network arg. Only for 'train' mode.")
    parser.add_argument('--loss_network', default='vgg16', choices=['vgg16', 'vgg19', 'resnet18'], help="The loss network used for perceptual loss computing. Only for 'train' mode")
    opt = parser.parse_args()

    if opt.mode == 'meta_train':
        meta_train(train_path=opt.train_folder, valid_path=opt.valid_folder, batch_size=opt.batch_size, epoch_nb=opt.epoch_nb, learning_rate=opt.learning_rate, meta_learning_rate=opt.meta_learning_rate, save_path=opt.save_weights, verbose=opt.verbose, weights_load=opt.load_weights, loss_func=opt.loss, loss_network=opt.loss_network, network=opt.network_name)
    elif opt.mode == 'finetuner_train':
        finetuner_train() # ToDo
    elif opt.mode == 'meta_upscale':
        MAMLupscale(in_path=opt.input, out_path=opt.output, weights_path=opt.load_weights, learning_rate=opt.learning_rate, batch_size=opt.batch_size, verbose=opt.verbose, device_name=opt.device, network=opt.network_name)
    elif opt.mode == 'evaluation':
        pass
    elif opt.mode == 'upscale_video':
        pass

    else:
        raise Exception("Invalid mode. Run this command if you need help: $ python run.py --help")
