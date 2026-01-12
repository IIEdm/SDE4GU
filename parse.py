def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def parser_add_main_args(parser):
    
    parser.add_argument('--adjoint', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--method', type=str, default='euler', choices=('euler', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-1)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--time', type=float, default=1.0)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--input_dropout', type=float, default=0.0, help='Input dropout rate.')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')

    parser.add_argument('--dataset', type=str, default='amazon-computer')
    parser.add_argument('--ood_type', type=str, default='label', choices=['label', 'feature'])
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_prop', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.1,
                        help='validation label proportion')
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=200)
    
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--samples', type=int, default=1)

    

    # training
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    


