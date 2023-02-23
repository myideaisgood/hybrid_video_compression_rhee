import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys
sys.path.append('.')
from config import parse_args
from utils.helpers import *

class Train_Dataset(Dataset):
    def __init__(self, args):

        self.qp = args.qp
        self.crop_size = args.crop_size

        self.data_dir = args.data_dir
        self.raw_data_dir = os.path.join(args.data_dir, args.train_dataset)
        self.dataset = args.train_dataset
        self.vvc_dir = args.vvc_dir

        self.reference_type = args.ReferenceType

        self.video_names = sorted([name for name in os.listdir(self.raw_data_dir) if os.path.isdir(os.path.join(self.raw_data_dir, name))])

        # Count frame numbers
        self.frame_num_list, self.frame_num = self.count_frame_num(self.video_names)
    
        self.transform = transforms.ToTensor()

    def count_frame_num(self, video_names):
        
        frame_num_list = []
        frame_num = 0
        for name in video_names:
            frame_num = len(os.listdir(os.path.join(self.raw_data_dir, name)))

            frame_num_list.append(frame_num)
        
        frame_num = sum(frame_num_list)
        
        return frame_num_list, frame_num

    def __len__(self):
        return self.frame_num
    
    def get_videoidx(self, idx):

        video_idx = 0

        for frame_num in self.frame_num_list:
            if idx - frame_num < 0 :
                break
            else:
                idx = idx - frame_num
                video_idx +=1

        frame_idx = idx

        return video_idx, frame_idx

    def crop_location(self, name):

        img = cv2.imread(name)

        H, W, _ = img.shape

        x = np.random.randint(0, W-self.crop_size+1)
        y = np.random.randint(0, H-self.crop_size+1)

        return x,y

    def read2tensor_crop(self, name, crop_x, crop_y):

        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        img = img[crop_y:crop_y+self.crop_size, crop_x:crop_x+self.crop_size,:]
        img = self.transform(img)

        return img    

    def __getitem__(self, idx):

        video_idx, frame_idx = self.get_videoidx(idx)

        frame_list = sorted(os.listdir(os.path.join(self.raw_data_dir, self.video_names[video_idx])))

        if self.reference_type == 'GT':
            ref_idx = frame_idx
        elif self.reference_type == 'prev':
            ref_idx = max(frame_idx-1,0)
        else:
            ref_idx = np.random.randint(len(frame_list))

        # Get image path
        x_t = os.path.join(self.raw_data_dir, self.video_names[video_idx], frame_list[frame_idx])
        x_ref = os.path.join(self.raw_data_dir, self.video_names[video_idx], frame_list[ref_idx])

        VVC_DIR = os.path.join(self.data_dir, self.vvc_dir, self.dataset, self.video_names[video_idx], str(self.qp))
        vvc_frame_list = sorted(os.listdir(VVC_DIR))
        c_t = os.path.join(VVC_DIR, vvc_frame_list[frame_idx])

        # Read cropped image
        crop_x, crop_y = self.crop_location(x_t)
        x_t = self.read2tensor_crop(x_t, crop_x, crop_y)
        x_ref = self.read2tensor_crop(x_ref, crop_x, crop_y)

        c_t = self.read2tensor_crop(c_t, crop_x, crop_y)

        return c_t, x_t, x_ref


class Eval_Dataset(Dataset):
    def __init__(self, args):

        self.qp = args.qp

        self.data_dir = args.data_dir
        self.raw_data_dir = os.path.join(args.data_dir, args.eval_dataset)
        self.dataset = args.eval_dataset
        self.vvc_dir = args.vvc_dir

        self.video_names = sorted([name for name in os.listdir(self.raw_data_dir) if os.path.isdir(os.path.join(self.raw_data_dir, name))])

        # Step 1. Count frame numbers
        self.frame_num_list, self.frame_num = self.count_frame_num(self.video_names)

    def __len__(self):
        return len(self.video_names)

    def count_frame_num(self, video_names):
        
        frame_num_list = []
        frame_num = 0
        for name in video_names:
            frame_num = len(os.listdir(os.path.join(self.raw_data_dir, name)))

            frame_num_list.append(frame_num)
            frame_num += frame_num
        
        return frame_num_list, frame_num

    def __getitem__(self, idx):

        RAW_DIR = os.path.join(self.raw_data_dir, self.video_names[idx])
        VVC_DIR = os.path.join(self.data_dir, self.vvc_dir, self.dataset, self.video_names[idx], str(self.qp))

        raw_frames = sorted(os.listdir(RAW_DIR))
        vvc_frames = sorted(os.listdir(VVC_DIR))

        raw_frames = [os.path.join(RAW_DIR, name) for name in raw_frames]
        vvc_frames = [os.path.join(VVC_DIR, name) for name in vvc_frames]

        return vvc_frames, raw_frames, self.video_names[idx]

if __name__ == '__main__':

    args = parse_args()

    SAVE_NUM = 20

    GPU_NUM = args.gpu_num
    QP = args.qp
    PLAYGROUND = 'playground/'
    BATCH_SIZE = args.batch_size

    DO_TRAIN = False
    DO_EVAL = True

    os.makedirs(PLAYGROUND, exist_ok=True)

    if DO_TRAIN:
        train_dataset = Train_Dataset(args)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True
        )    

        save_idx = 0

        for i, data in enumerate(train_dataloader):

            c_t, x_t, x_ref = data

            for b_idx in range(BATCH_SIZE):
                cur_c_t, cur_x_t, cur_x_ref = c_t[b_idx], x_t[b_idx], x_ref[b_idx]

                cur_c_t = tensor2img(cur_c_t)
                cur_x_t = tensor2img(cur_x_t)
                cur_x_ref = tensor2img(cur_x_ref)

                save_img = np.concatenate([cur_c_t, cur_x_t, cur_x_ref], axis=1)

                outname = PLAYGROUND + 'train_qp_' + str(QP) +  '_img_' + str(i).zfill(3) + '_' + str(b_idx).zfill(1) + '.jpg'

                cv2.imwrite(outname, save_img)

                save_idx += 1

            if save_idx >= SAVE_NUM:
                sys.exit(1)

    elif DO_EVAL:
        eval_dataset = Eval_Dataset(args)

        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        save_idx = 0

        for i, data in enumerate(eval_dataloader):

            vvc_frames, raw_frames, videoname = data