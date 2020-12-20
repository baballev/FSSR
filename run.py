"""CLI runner file."""
def main(opt):
    from FSSR import VanillaTrain, MetaTrain, Test
    
    if opt.mode == 'vanilla':
        #require('train_folder', 'valid_folders', 'scale', 'batch_size', 'epochs', 'learning_rate')
        VanillaTrain(train_fp=opt.train_folder, valid_fps=opt.valid_folders, 
            load=opt.load_weights, scale=opt.scale, bs=opt.batch_size, lr=opt.learning_rate, 
            loss=opt.loss, size=opt.resize, epochs=opt.epochs, wandb=wandb)
    
    elif opt.mode == 'meta':
        run = MetaTrain(opt) 
        run(debug=opt.debug)

    elif opt.mode == 'test':
        #require('valid_folders', 'models', 'scale', 'nb_shots', 'update_test_steps', 'learning_rate')
        run = Test(model_fps=opt.models, test_fp=opt.valid_folders[0], scale=opt.scale,
            shots=opt.nb_shots, lr=opt.learning_rate, size=opt.resize, loss=opt.loss, 
            update_steps=opt.update_test_steps, wandb=wandb)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('mode', choices=['vanilla', 'meta', 'test'],
        help="The name of the mode to run")
    parser.add_argument('--n-resblocks', type=int, default=16,
        help='Number of residual blocks in EDSR model.')    
    parser.add_argument('--n-feats', type=int, default=64,
        help='Number of feature maps in EDSR model.')
    parser.add_argument('--nb-tasks', type=int,
        help='Number of sampled tasks per outer loop updates in meta training.')
    parser.add_argument('--batch-size', type=int,
        help='The number of images for each training iteration as an integer.')
    parser.add_argument('--scale', type=int,
        help='The scaling factor for an upscaling task.')
    parser.add_argument('--epochs', type=int,
        help='Number of epochs for training i.e number of times the training set is iterated over.')
    parser.add_argument('--shots', type=int,
        help='Number of shots in each task.')
    parser.add_argument('--loss', default='L1', choices=['VGG', 'L1', 'L2'],
        help='Loss function used for training the model.')
    parser.add_argument('--train-set',
        help='Dataset preset name name or path training set directory.')
    parser.add_argument('--valid-sets', nargs='+', type=str,
        help='Dataset preset name name or path validation set directory.')
    parser.add_argument('--clusters', type=str,
        help='File containing clusters for a dataset.')
    parser.add_argument('--load', type=str, default=False,
        help='Path to the weight file for finetuning.')
    parser.add_argument('--lr', type=float,
        help='Learning rate for training.')
    parser.add_argument('--meta-lr', type=float,
        help='Learning rate of the meta training (inner loop)')
    parser.add_argument('--lr-annealing', type=float, default=False,
        help='Max number of iterations until learning rate reaches zero. No decay if set to None.' \
             'Defined as a fraction of the number of batch in a epoch i.e. len(train_dl).')
    parser.add_argument('--weight-decay', type=float, default=0,
        help='Training L2 weight decay.')
    parser.add_argument('--update-steps', type=int,
        help='For meta-learning: number of gradient updates when finetuning during training.')
    parser.add_argument('--update-test-steps', type=int,
        help='For meta-learning: number of gradient updates when finetuning during validation.')
    parser.add_argument('--wandb', type=str, 
        help='Name of the wandb project to save the run to.')
    parser.add_argument('--no-wandb', action='store_true', default=False,
        help='Flag to disable wandb recording completely.')
    parser.add_argument('--size', nargs='+', type=int,
        help='Image size input of the SR model as (h, w).')
    parser.add_argument('--models', nargs='+', type=str,
        help='List of models meta/non-meta to evaluate.')
    parser.add_argument('--debug', action='store_true', default=False,
        help='Will prevent run from starting and print a summary of the options.')

    opt = parser.parse_args()
    opt.wandb = False if opt.no_wandb else opt.wandb
    assert len(opt.size) == 0 or len(opt.size) == 2

    main(opt)
