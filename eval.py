import cv2
import os
from tqdm import tqdm
import logging

from torchvision.transforms import transforms

from config import parse_args
from model import HochangNet
from utils.helpers import *
from utils.data_loaders import *

def read_frames(frame_idx, hevc_frames, raw_frames, ReferenceType):

    c_t = hevc_frames[frame_idx][0]
    x_t = raw_frames[frame_idx][0]

    if ReferenceType == 'GT':
        x_ref = raw_frames[frame_idx][0]
    elif ReferenceType == 'prev':
        x_ref = raw_frames[max(frame_idx-1,0)][0]
    else:
        x_ref = raw_frames[0][0]
        c_ref = hevc_frames[0][0]

    c_t = cv2.cvtColor(cv2.imread(c_t), cv2.COLOR_BGR2RGB)
    x_t = cv2.cvtColor(cv2.imread(x_t), cv2.COLOR_BGR2RGB)
    x_ref = cv2.cvtColor(cv2.imread(x_ref), cv2.COLOR_BGR2RGB)

    return c_t, x_t, x_ref

def split_data(datas, split_num=4):

    output = []

    for idx, data in enumerate(datas):

        [_,_,H,W] = data.size()

        if split_num == 1:
            output.append([data])
        
        elif split_num == 4:

            half_H, half_W = int(H/2), int(W/2)

            data_lu = data[:,:,:half_H, :half_W]
            data_ru = data[:,:,:half_H, half_W:]
            data_ld = data[:,:,half_H:, :half_W]
            data_rd = data[:,:,half_H:, half_W:]

            output.append([data_lu, data_ru, data_ld, data_rd])

        elif split_num == 9:

            H1, H2, H3 = int(H/3), int(2*H/3), H
            W1, W2, W3 = int(W/3), int(2*W/3), W

            data1 = data[:,:,0:H1, 0:W1]
            data2 = data[:,:,0:H1, W1:W2]
            data3 = data[:,:,0:H1, W2:W3]

            data4 = data[:,:,H1:H2, 0:W1]
            data5 = data[:,:,H1:H2, W1:W2]
            data6 = data[:,:,H1:H2, W2:W3]

            data7 = data[:,:,H2:H3, 0:W1]
            data8 = data[:,:,H2:H3, W1:W2]
            data9 = data[:,:,H2:H3, W2:W3]

            output.append([data1, data2, data3, data4, data5, data6, data7, data8, data9])

        else:
            raise NotImplementedError

    return output

def img2cuda(imgs, device):

    for idx, img in enumerate(imgs):
        imgs[idx] = var_or_cuda(img, device=device)
    
    return imgs

def convert2tensor(imgs, transform):

    for idx, img in enumerate(imgs):
        img = transform(img)
        imgs[idx] = torch.unsqueeze(img, dim=0)

    return imgs

def tensor2img(img_tensor):
    img_tensor = img_tensor[0].permute(1,2,0)
    img_np = img_tensor.detach().cpu().numpy()
    img_np = (255*img_np).astype(np.uint8)

    return img_np

def unite_data(datas, SPLIT_EVAL_NUM):

    H, W, _ = datas[0].shape

    if SPLIT_EVAL_NUM == 1:
        output = np.zeros([H, W, 3], dtype=np.uint8)
        output = datas[0]

    elif SPLIT_EVAL_NUM == 4:

        output = np.zeros([2*H, 2*W, 3], dtype=np.uint8)

        output[:H, :W, :] = datas[0]
        output[:H, W:, :] = datas[1]

        output[H:, :W, :] = datas[2]
        output[H:, W:, :] = datas[3]


    elif SPLIT_EVAL_NUM == 9:
        output = np.zeros([3*H, 3*W, 3], dtype=np.uint8)

        output[:H, :W, :] = datas[0]
        output[:H, W:2*W, :] = datas[1]
        output[:H, 2*W:3*W, :] = datas[2]

        output[H:2*H, 0:W, :] = datas[3]
        output[H:2*H, W:2*W, :] = datas[4]
        output[H:2*H, 2*W:3*W, :] = datas[5]

        output[2*H:3*H, 0:W, :] = datas[6]
        output[2*H:3*H, W:2*W, :] = datas[7]
        output[2*H:3*H, 2*W:3*W, :] = datas[8]

    return output

def evaluate(network, dataloader, device, logging, TRAIN_STEP, SPLIT_EVAL_NUM, ReferenceType):

    transform = transforms.ToTensor()

    network.eval()

    total_psnr_list = []

    with torch.no_grad():

        for idx, data in enumerate(tqdm(dataloader)):

            hevc_frames, raw_frames, videoname = data

            psnr_list = []

            for frame_idx in range(len(hevc_frames)):

                c_t, x_t, x_ref = read_frames(frame_idx, hevc_frames, raw_frames, ReferenceType)

                x_t_np = x_t

                [c_t, x_t, x_ref] = convert2tensor([c_t, x_t, x_ref], transform)
                [c_ts, x_ts, x_refs] = split_data([c_t, x_t, x_ref], SPLIT_EVAL_NUM)

                frame_psnr = []

                preds = []

                for split_idx, (c_t_split, x_t_split, x_ref_split) in enumerate(zip(c_ts, x_ts, x_refs)):

                    [c_t_split, x_t_split, x_ref_split] = img2cuda([c_t_split, x_t_split, x_ref_split], device)

                    pred_1, pred_2 = network(c_t=c_t_split, x_ref=x_ref_split)

                    if TRAIN_STEP == 'step1':
                        pred = pred_1
                    else:
                        pred = pred_2

                    pred = pred_2

                    pred = torch.clip(pred, min=0.0, max=1.0)
                    preds.append(tensor2img(pred))

                prediction = unite_data(preds, SPLIT_EVAL_NUM)

                frame_psnr = get_psnr_np(prediction, x_t_np)

                psnr_list.append(frame_psnr)
                total_psnr_list.append(frame_psnr)
            
            video_psnr = sum(psnr_list) / len(psnr_list)

            logging.info('%s : %.2f dB' % (videoname[0], video_psnr))
        
        eval_psnr = sum(total_psnr_list) / len(total_psnr_list)

        return eval_psnr


def do_eval():
    args = parse_args()

    GPU_NUM = args.gpu_num
    NUM_WORKERS = args.num_workers

    TRAIN_STEP = args.train_step
    EXP_DIR = 'experiments/' + args.exp_name
    WEIGHTS = TRAIN_STEP + '_' +  args.weights

    SPLIT_EVAL_NUM = args.split_eval_num
    REFERENCE_TYPE = args.ReferenceType

    # Check if directory does not exist
    os.makedirs('experiments/', exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    # Set up logger
    filename = os.path.join(EXP_DIR, 'logs_eval.txt')
    logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))
        logging.info('\t%15s:\t%s' % (key, value))

    # Set up Dataset
    eval_dataset = Eval_Dataset(args)

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

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        network.to(device)

    # Load the pretrained model
    logging.info('Recovering from %s ...' % os.path.join(EXP_DIR, WEIGHTS))
    checkpoint = torch.load(os.path.join(EXP_DIR, WEIGHTS))
    network.load_state_dict(checkpoint['network'])
    logging.info('Recover completed.')

    # Evaluate
    eval_psnr = evaluate(network, eval_dataloader, device, logging, TRAIN_STEP, SPLIT_EVAL_NUM, REFERENCE_TYPE)
    logging.info('EVAL PSNR = %.2f' % (eval_psnr))


if __name__ == '__main__':
    do_eval()    