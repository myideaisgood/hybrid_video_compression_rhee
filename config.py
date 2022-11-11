import argparse
from random import choices

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """

    parser.add_argument('--crf', type=int, default=25, help='crf')
    parser.add_argument('--train_step', type=str, default='step2', help='step1, step2')

    # Session parameters
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Minibatch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Worker')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--print_every', type=int, default=1, help='How many iterations print for loss evaluation')

    parser.add_argument('--split_eval_num', type=int, default=9, help='Number to split during inference (due to memory size) (Can be 1, 4, 9)')

    ### learning rate
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0005')

    # Directory parameters
    parser.add_argument('--data_dir', type=str, default='../DATASET/', help='dataset directory')
    parser.add_argument('--train_dataset', type=str, default='MCL-JCV/', help='train dataset')
    parser.add_argument('--eval_dataset', type=str, default='UVG/', help='evaluation dataset')
    parser.add_argument('--exp_name', type=str, default='UVG_crf25/', help='Experiment Name directory')
    parser.add_argument('--weights', type=str, default="ckpt.pth", help='Weight Name')
    parser.add_argument('--hevc_dir', type=str, default='hevc_result/', help='hevc result directory')

    # Data Crop parameters
    parser.add_argument('--crop_size', type=int, default=256, help='Crop Size')

    # Architecture parameters
    parser.add_argument('--step1_channels', type=int, nargs='+', default=[64, 96, 128], help='Channel list')
    parser.add_argument('--step2_enc_channels', type=int, nargs='+', default=[64, 96, 128], help='Channel list')
    parser.add_argument('--step2_dec_channels', type=int, nargs='+', default=[128, 128], help='Channel list')
    parser.add_argument('--deform_enc_channels', type=int, nargs='+', default=[64, 64, 96, 96], help='Channel list')
    
    parser.add_argument('--ReferenceType', type=str, default='first', help='GT, prev, first')

    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--activation_type', type=str, default='gelu', choices=['relu', 'swish', 'gelu'])
    parser.add_argument('--img_out_type', type=str, default='tanh', choices=['tanh', 'sigmoid', 'none'])

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))
