"""CLI runner file."""
def main(opt, require):
    from FSSR import VanillaTrain, MetaTrain, Test

    if opt.mode == 'vanilla-train':
        require('train_folder', 'valid_folders', 'scale', 'batch_size', 'epochs', 'learning_rate')
        run = VanillaTrain(train_fp=opt.train_folder, valid_fps=opt.valid_folders, load=opt.load_weights,
            scale=opt.scale, bs=opt.batch_size, lr=opt.learning_rate, loss=opt.loss, size=opt.resize,
            wandb=not opt.no_wandb)

        run(epochs=opt.epochs)

    elif opt.mode == 'meta-train':
        require('train_folder', 'clusters', 'scale', 'batch_size', 'epochs', 'nb_shots',
            'update_steps', 'update_test_steps', 'learning_rate', 'meta_learning_rate')
        run = MetaTrain(dataset_fp=opt.train_folder, clusters_fp=opt.clusters, load=opt.load_weights,
            scale=opt.scale, shots=opt.nb_shots, nb_tasks=opt.batch_size, lr=opt.learning_rate,
            meta_lr=opt.meta_learning_rate, size=opt.resize, loss=opt.loss,
            lr_annealing=opt.lr_annealing, wandb=not opt.no_wandb)

        run(epochs=opt.epochs, update_steps=opt.update_steps, update_test_steps=opt.update_test_steps)

    elif opt.mode == 'models-test':
        require('valid_folders', 'models', 'scale', 'nb_shots', 'update_test_steps', 'learning_rate')
        run = Test(model_fps=opt.models, test_fp=opt.valid_folders[0], scale=opt.scale,
            shots=opt.nb_shots, lr=opt.learning_rate, size=opt.resize, loss=opt.loss, wandb=not opt.no_wandb)

        run(update_steps=opt.update_test_steps)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils import require_args

    parser = ArgumentParser()

    parser.add_argument('mode', choices=['vanilla-train', 'meta-train', 'models-test'],
        help="The name of the mode to run")
    parser.add_argument('--batch-size', type=int,
        help='The number of images for each training iteration as an integer.')
    parser.add_argument('--scale', type=int,
        help='The scaling factor for an upscaling task.')
    parser.add_argument('--epochs', type=int,
        help='Number of epochs for training i.e number of times the training set is iterated over.')
    parser.add_argument('--nb-shots', type=int,
        help='Number of shots in each task.')
    parser.add_argument('--loss', default='L1', choices=['VGG', 'L1', 'L2'],
        help='Loss function used for training the model.')
    parser.add_argument('--train-folder',
        help='Dataset preset name name or path training set directory.')
    parser.add_argument('--valid-folders', nargs='+', type=str,
        help='Dataset preset name name or path validation set directory.')
    parser.add_argument('--clusters', type=str,
        help='File containing clusters for a dataset.')
    parser.add_argument('--load-weights',
        help='Path to the weight file for finetuning.')
    parser.add_argument('--learning-rate', type=float,
        help='Learning rate for training.')
    parser.add_argument('--meta-learning-rate', type=float,
        help='Learning rate of the meta training.')
    parser.add_argument('--lr-annealing', type=float,
        help='Max number of iterations until learning rate reaches zero. No decay if set to None. As fraction of training set size.')
    parser.add_argument('--update-steps', type=int,
        help='For meta-learning: number of gradient updates when finetuning during training.')
    parser.add_argument('--update-test-steps', type=int,
        help='For meta-learning: number of gradient updates when finetuning during validation.')
    parser.add_argument('--no-wandb', action="store_true", default=False,
        help='Whether or not to record the run with wandb.')
    parser.add_argument('--resize', nargs='+', type=int, default=(256, 512),
        help='Image size of the model input as (h, w).')
    parser.add_argument('--models', nargs='+', type=str,
        help='List of models meta/non-meta to evaluate.')

    opt = parser.parse_args()
    require = require_args(opt)

    assert len(opt.resize) == 0 or len(opt.resize) == 2

    main(opt, require)
