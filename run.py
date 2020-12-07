"""CLI runner file."""
def main(opt, require):
    # from FSSR import vanilla_train, meta_train, models_test, finetuneMaml, MAMLupscale, upscale

    # 1) revisit everything so we have wandb logging + loss output in file + weights saved in fs
    # 2) train DIV2Kx2 (validation w/ w/out styles) train DIV2Kx2 styles (validation w/ w/out styles)
    # 3) train a meta network: how many epochs??

    from FSSR import VanillaTrain, MetaTrain

    if opt.mode == 'vanilla-train':
        require('train_folder', 'valid_folders', 'scale', 'batch_size', 'epochs')
        run = VanillaTrain(train_fp=opt.train_folder, valid_fps=opt.valid_folders, load=opt.load_weights,
            scale=opt.scale, bs=opt.batch_size, lr=opt.learning_rate, loss=opt.loss, wandb=not opt.no_wandb)

        run(epochs=opt.epochs)

    if opt.mode == 'meta-train':
        require('train_folder', 'valid_folders', 'scale', 'batch_size', 'epochs')
        run = MetaTrain(train_fp=opt.train_folder, valid_fps=opt.valid_folders, load=opt.load_weights,
            scale=opt.scale, shots=opt.nb_shots, nb_tasks=opt.batch_size, lr=opt.learning_rate,
            meta_lr=opt.meta_learning_rate, size=opt.resize, loss=opt.loss, wandb=not opt.no_wandb)

        run(epochs=opt.epochs, update_steps=opt.update_steps, update_test_steps=opt.update_test_steps)

    """if opt.mode == 'vanilla-train':
        require('train_folder', 'valid_folders', 'epochs', 'batch_size', 'scale')
        summarize('train_folder', 'valid_folders', 'epochs', 'scale', 'batch_size', 'load_weights')
        vanilla_train(train_fp=opt.train_folder, valid_fps=opt.valid_folders, load=opt.load_weights,
            scale=opt.scale, bs=opt.batch_size, epochs=opt.epochs, lr=opt.learning_rate)

    elif opt.mode == 'meta-train':
        require('train_folder', 'valid_folders', 'epochs', 'batch_size', 'scale')
        summarize('train_folder', 'valid_folders', 'epochs', 'scale', 'batch_size', 'load_weights')
        meta_train(train_fp=opt.train_folder, valid_fp=opt.valid_folders[0], load=opt.load_weights,
            scale=opt.scale, shots=opt.nb_shots, bs=opt.batch_size, epochs=opt.epochs,
            lr=opt.learning_rate, meta_lr=opt.meta_learning_rate)

    elif opt.mode == 'models-test':
        require('valid_folders', 'models', 'scale', 'nb_shots')
        models_test(test_fp=opt.valid_folders[0], model_fps=opt.models, scale=opt.scale,
            shots=opt.nb_shots, lr=opt.learning_rate, epochs=opt.epochs)

    elif opt.mode == 'finetune-maml':
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

    elif opt.mode == 'meta-upscale':
        MAMLupscale(in_path=opt.input,
                    out_path=opt.output,
                    weights_path=opt.load_weights,
                    learning_rate=opt.learning_rate,
                    batch_size=opt.batch_size,
                    verbose=opt.verbose,
                    device_name=opt.device,
                    network=opt.model)

    elif opt.mode == 'evaluation':
        from evaluation import evaluation
        evaluation(in_path=opt.input, out_path=opt.output, verbose=True)

    elif opt.mode == 'upscale-video':
        pass

    elif opt.mode == 'upscale':
        upscale(load_weights=opt.load_weights, input=opt.input, out=opt.output)"""


if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils import require_args

    parser = ArgumentParser()

    parser.add_argument('mode', choices=['vanilla-train', 'meta-train', 'models-test', \
        'meta-upscale', 'finetune-maml', 'evaluation', 'upscale-video', 'upscale'],
        help="The name of the mode to run")

    parser.add_argument('--batch-size', type=int,
        help='The number of images for each training iteration as an integer.')
    parser.add_argument('--scale', type=int,
        help='The scaling factor for an upscaling task.')
    parser.add_argument('--epochs', type=int,
        help='Number of epochs for training i.e number of times the training set is iterated over.')
    parser.add_argument('--nb-shots', default=10, type=int,
        help='Number of shots in each task.')
    parser.add_argument('--loss', default='L1', choices=['VGG', 'L1', 'L2'],
        help='Loss function used for training the model.')
    parser.add_argument('--train-folder',
        help='Dataset preset name name or path training set directory.')
    parser.add_argument('--valid-folders', nargs='+', type=str,
        help='Dataset preset name name or path validation set directory.')
    parser.add_argument('--load-weights',
        help='Path to the weight file for finetuning.')
    parser.add_argument('--learning-rate', default=0.001, type=float,
        help='Learning rate for training.')
    parser.add_argument('--meta-learning-rate', default=0.0001, type=float,
        help='Learning rate of the meta training.')
    parser.add_argument('--update-steps', default=10, type=int,
        help='For meta-learning: number of gradient updates when finetuning during training.')
    parser.add_argument('--update-test-steps', default=10, type=int,
        help='For meta-learning: number of gradient updates when finetuning during validation.')
    parser.add_argument('--no-wandb', action="store_true", default=False,
        help='Whether or not to record the run with wandb.')
    parser.add_argument('--resize', nargs='+', type=int, default=(256, 512),
        help='Image size of the model input as (h, w).')

    # ~todo
    parser.add_argument('--models', nargs='+', type=str)
    parser.add_argument('--save-weights',
        help='Path to save the weights after training (.pt).')
    parser.add_argument('--input', default='../CelebA/', # '../../../../mnt/data/prim_project/dataset/FSSR/CelebA/'
        help="Path to the directory containing the images to upscale. Only used for upscaling or evaluation.")
    parser.add_argument('--output', default='./out/',
        help="Destination directory for the benchmark in 'evaluation' mode or upscaled images in 'upscale' mode.")
    parser.add_argument('--verbose', default=True, type=bool, choices=[True, False],
        help="Whether the script print info in stdout.")
    parser.add_argument('--loss_network', default='vgg16', choices=['vgg16', 'vgg19', 'resnet18'],
        help="The loss network used for perceptual loss computing. Only for 'train' mode")
    parser.add_argument('--finetune_depth', default=0, type=int,
        help="Number of parameter tensors to be finetuned in finetune_maml mode. 0 to modify all layers.")

    opt = parser.parse_args()
    require = require_args(opt)

    assert len(opt.resize) == 0 or len(opt.resize) == 2

    main(opt, require)
