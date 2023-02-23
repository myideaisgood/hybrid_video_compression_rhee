import os
import logging
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from config import parse_args
from model import HochangNet
from utils.data_loaders import Train_Dataset, Eval_Dataset
from utils.helpers import *
from eval import *

def train():
    
    args = parse_args()

    GPU_NUM = args.gpu_num
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    EPOCHS = args.epochs
    PRINT_EVERY = args.print_every
    START_EVERY = args.start_every
    EVAL_EVERY = args.eval_every

    REFERENCE_TYPE = args.ReferenceType
    LOSS_TYPE = args.loss_type

    LR = args.lr

    TRAIN_STEP = args.train_step
    SPLIT_EVAL_NUM = args.split_eval_num

    EXP_DIR = 'experiments/' + args.exp_name
    WEIGHTS = TRAIN_STEP + '_' +  args.weights

    # Check if directory does not exist
    os.makedirs('experiments/', exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    # Set up logger
    filename = os.path.join(EXP_DIR, 'logs_' + TRAIN_STEP + '.txt')
    logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))
        logging.info('\t%15s:\t%s' % (key, value))

    # Set up Dataset
    train_dataset = Train_Dataset(args)
    eval_dataset = Eval_Dataset(args)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # Set up Network
    network = HochangNet(args)
    network.apply(initialize_weights)

    logging.info('Network Parameters : %.1f M' % (count_parameters(network) * 10**(-6)))

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        network.to(device)

    # Load the pretrained model if exists
    init_epoch = 0
    best_metric = 0.0

    if TRAIN_STEP == 'step1':
        if os.path.exists(os.path.join(EXP_DIR, WEIGHTS)):
            logging.info('Recovering from %s ...' % os.path.join(EXP_DIR, WEIGHTS))
            checkpoint = torch.load(os.path.join(EXP_DIR, WEIGHTS))
            init_epoch = checkpoint['epoch_idx']
            best_metric = checkpoint['best_metric']
            network.load_state_dict(checkpoint['network'])
            logging.info('Recover completed. Current epoch = #%d' % (init_epoch))
    
    elif TRAIN_STEP == 'step2':
        if os.path.exists(os.path.join(EXP_DIR, WEIGHTS)):
            logging.info('Recovering from %s ...' % os.path.join(EXP_DIR, WEIGHTS))
            checkpoint = torch.load(os.path.join(EXP_DIR, WEIGHTS))
            init_epoch = checkpoint['epoch_idx']
            best_metric = checkpoint['best_metric']
            network.load_state_dict(checkpoint['network'])
            logging.info('Recover completed. Current epoch = #%d' % (init_epoch))
        else:
            WEIGHTS_TEMP =  'step1_' +  args.weights
            logging.info('Recovering from %s ...' % os.path.join(EXP_DIR, WEIGHTS_TEMP))
            checkpoint = torch.load(os.path.join(EXP_DIR, WEIGHTS_TEMP))
            step1_best_metric = checkpoint['best_metric']
            network.load_state_dict(checkpoint['network'])
            logging.info('Recover completed. Best Metric of Step 1 = %.2f' % (step1_best_metric))   

    # Criterion
    if LOSS_TYPE == 'l2':
        criterion = nn.MSELoss()
    elif LOSS_TYPE == 'l1':
        criterion = nn.L1Loss()
    else:
        raise NotImplementedError

    # Create Optimizer / Scheduler
    optimizer = optim.AdamW(network.parameters(), lr=LR)

    # Freeze according to step
    if TRAIN_STEP == 'step1':
        network.freeze_step2()
    elif TRAIN_STEP == 'step2':
        network.freeze_step1()
    else:
        raise NotImplementedError

    # Check trainable parameters
    trainable_list, module_param_list, total_params = count_module_parameters(network)
    logging.info("********** Trainable Parameters **********")
    for idx in range(len(trainable_list)):
        logging.info("\t%15s : %.1f M" % (trainable_list[idx], module_param_list[idx] * 10**(-6)))

    logging.info("\t%15s : %.1f M" % ('Total',total_params * 10**(-6)))    

    # Train
    for epoch_idx in range(init_epoch+1, EPOCHS):

        network.train()

        train_loss = 0

        # Iterate over dataset
        for i, data in enumerate(tqdm(train_dataloader)):
            
            c_t, x_t, x_ref = data
            [c_t, x_t, x_ref] = img2cuda([c_t, x_t, x_ref], device)

            pred_1, pred_2 = network(c_t=c_t, x_ref=x_ref)

            if TRAIN_STEP == 'step1':
                loss = criterion(pred_1, x_t)
            elif TRAIN_STEP == 'step2':
                loss = criterion(pred_2, x_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * BATCH_SIZE

        train_loss /= len(train_dataset)

        if epoch_idx % PRINT_EVERY == 0:
            logging.info('Epoch [%d/%d] Loss = %.5f LR = %.7f' % (epoch_idx, EPOCHS, train_loss, LR))

        if epoch_idx % EVAL_EVERY == 0 and epoch_idx > START_EVERY:
            eval_psnr = evaluate(network, eval_dataloader, device, logging, TRAIN_STEP, SPLIT_EVAL_NUM, REFERENCE_TYPE)

            if eval_psnr > best_metric:
                best_metric = eval_psnr

                # Save Network
                save_path = os.path.join(EXP_DIR, WEIGHTS)
                torch.save({
                    'epoch_idx': epoch_idx,
                    'best_metric': best_metric,
                    'network' : network.state_dict(),
                }, save_path)

                logging.info('Saved checkpoint to %s ...' % save_path)

            logging.info('EVAL PSNR = %.2f  BEST PSNR = %.2f' % (eval_psnr, best_metric))                

if __name__ == '__main__':
    train()    