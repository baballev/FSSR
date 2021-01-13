"""CLI runner file."""
def main(opt):
    from FSSR import VanillaTrain, MetaTrain, Test

    # wandb sweep metric is PSNR on test set (CelebA)    
    # meta_lr <-> lr...
    # TaskDataset __init__ refactor
    # superset the meta datasets since transform handling is the exact same on Task & Cluster

    # Christmas discoveries: 
    #  (1) use Adam like EDSR not SGD!!!
    #  (2) randomness in mini-batches 'task-batches' leads to more instability vs fixing the elements 
    #      on dataset init. Randomness also has the effect of mixing spt and qry (not a good idea).
    #      Randomness was used in ClusterDataset because clusters had a size > nb_shots.
    #      Tried with ClusterDataset160 min-cluster-size=5 and **fixed** query and got stability.
    # Post:
    # Sweep with 48x48. Runs can be very unstable in random parts of the training (related w/ the
    # fact that support is picked at random in the set). To the point that gradients explode. This
    # is usually seen when lr is rather high, nb_tasks is low (<5) and here (shots=2).
    if opt.mode == 'vanilla':
        VanillaTrain(opt)()
    
    elif opt.mode == 'meta':
        MetaTrain(opt)()

    elif opt.mode == 'test':
        Test(opt)()

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
    parser.add_argument('--epochs', type=int, default=100,
        help='Number of epochs for training i.e number of times the training set is iterated over.')
    parser.add_argument('--timesteps', type=int, default=False,
        help='Number of updates to perform, will overwrite --epochs if set based on len(train_dl).')
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
    parser.add_argument('--first-order', type=bool, default=True,
        help='Whether to perform First Order MAML (will saves memory!)')
    parser.add_argument('--lr', type=float,
        help='Learning rate for training.')
    parser.add_argument('--meta-lr', type=float,
        help='Learning rate of the meta training (inner loop)')
    parser.add_argument('--lr-annealing', type=float, default=False,
        help='Number of epochs until learning rate reaches zero. No decay if set to None.')
    parser.add_argument('--weight-decay', type=float, default=0,
        help='Training L2 weight decay.')
    parser.add_argument('--update-steps', type=int,
        help='For meta-learning: number of gradient updates when finetuning during training.')
    parser.add_argument('--update-test-steps', type=int,
        help='For meta-learning: number of gradient updates when finetuning during validation.')
    parser.add_argument('--wandb', type=str, default=False,
        help='Name of the wandb project to save the run to. Default is set to no wandb recording.')
    parser.add_argument('--size', nargs='+', type=int,
        help='Image size input of the SR model as (h, w).')
    parser.add_argument('--models', nargs='+', type=str,
        help='List of models meta/non-meta to evaluate.')
    parser.add_argument('--debug', action='store_true', default=False,
        help='Will prevent run from starting and print a summary of the options.')

    opt = parser.parse_args()
    assert not opt.size or (opt.size and (len(opt.size) == 0 or len(opt.size) == 2))

    main(opt)
