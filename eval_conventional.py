import os
import cv2
import numpy as np
import math
import logging

def do_eval(data_dir, dataset, crfs, FRAME_NAME):


    HEVC_DIR = 'hevc_result/'
    
    # Create ../DATASET/hevc_result/dataset/
    # Example : ../DATASET/hevc_result/UVG/
    os.makedirs(os.path.join(data_dir, HEVC_DIR), exist_ok=True)
    os.makedirs(os.path.join(data_dir, HEVC_DIR, dataset), exist_ok=True)

    DATA_DIR = os.path.join(data_dir, dataset)

    logging.basicConfig(filename='log_conventional.txt',format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    logging.info('*** Start ***')

    video_names = sorted([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])

    crf_bpp_list = []
    crf_bpp_video_list = []
    crf_psnr_list = []
    crf_psnr_video_list = []

    for _ in range(len(video_names)):
        crf_bpp_video_list.append([])
        crf_psnr_video_list.append([])

    crf_idx = 0

    for crf in crfs:

        # Create ../DATASET/hevc_result/dataset/crf/
        os.makedirs(os.path.join(data_dir, HEVC_DIR, dataset, str(crf)), exist_ok=True)

        SAVE_DIR = os.path.join(data_dir, HEVC_DIR, dataset, str(crf))

        pixel_nums = 0
        video_bits = 0

        video_idx = 0

        crf_psnr = []

        # Do HEVC if needed
        for video_name in video_names:
            video_dir = os.path.join(DATA_DIR, video_name)
            outname = os.path.join(SAVE_DIR, video_name + '_' + str(crf) + '.mp4')

            # Do HEVC
            if not os.path.exists(outname):
                cur_video_dir = os.path.join(DATA_DIR, video_name)
                os.system('ffmpeg -i %s -c:v hevc -preset medium -x265-params bframes=0 -crf %d %s' % (os.path.join(cur_video_dir, FRAME_NAME), crf, outname))

            # Video to frames
            if not os.path.exists(os.path.join(SAVE_DIR, video_name)):
                os.makedirs(os.path.join(SAVE_DIR, video_name))
                save_name = os.path.join(SAVE_DIR, video_name, FRAME_NAME)
                os.system('ffmpeg -i %s %s' % (outname, save_name))

            # Calculate BPP
            frame_list = sorted(os.listdir(video_dir))
            frame_num = len(frame_list)
            frame0 = cv2.imread(os.path.join(video_dir, frame_list[0]))
            H,W,_ = frame0.shape

            filesize_bit = 8*os.stat(outname).st_size
            video_bits += filesize_bit

            pixel_nums += H*W*frame_num
            video_bpp = filesize_bit / (H*W*frame_num)

            # Calculate PSNR
            psnr_video = []
            for frame_idx in range(frame_num):
                ori_frame = cv2.imread(os.path.join(DATA_DIR, video_name, frame_list[frame_idx]))
                hevc_frame = cv2.imread(os.path.join(SAVE_DIR, video_name, frame_list[frame_idx]))

                psnr_cur = get_psnr_np(ori_frame, hevc_frame)

                psnr_video.append(psnr_cur)
                crf_psnr.append(psnr_cur)

            video_psnr = sum(psnr_video) / len(psnr_video)

            logging.info('Video Name : %s  CRF : %d  bpp : %.3f  PSNR : %.3f' % (video_name, crf, video_bpp, video_psnr))

            crf_bpp_video_list[crf_idx].append(video_bpp)
            crf_psnr_video_list[crf_idx].append(video_psnr)

        crf_bpp = video_bits / pixel_nums
        crf_psnr = sum(crf_psnr) / len(crf_psnr)

        logging.info('======= CRF %d    bpp %.3f   PSNR : %.3f ======' % (crf, crf_bpp, crf_psnr))

        crf_bpp_list.append(crf_bpp)
        crf_psnr_list.append(crf_psnr)

        crf_idx += 1

    for idx in range(len(crfs)):
        logging.info('CRF %d   BPP %.3f   PSNR %.2f' % (crfs[idx], crf_bpp_list[idx], crf_psnr_list[idx]))
    
    for video_idx in range(len(video_names)):
        logging.info('==== %s =====' % (video_names[video_idx]))
        for crf_idx in range(len(crfs)):
            logging.info('CRF %d (%.3f, %.2f)' % (crfs[crf_idx], crf_bpp_video_list[crf_idx][video_idx], crf_psnr_video_list[crf_idx][video_idx]))

    logging.info('*** End ***')    

def get_psnr_np(original, compressed):
    original = original.astype(np.float)
    compressed = compressed.astype(np.float)

    mse = np.mean((original - compressed)**2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='UNet')
    parser.add_argument('--crfs', type=int, nargs='+', default=[21, 22, 23, 24, 25, 26], help='crfs')
    parser.add_argument('--data_dir', type=str, default='../DATASET/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='MCL-JCV/', help='Dataset name')
    args = parser.parse_args()

    if 'vimeo' in args.dataset:
        FRAME_NAME = 'im%d.png'
    else:
        FRAME_NAME = 'f%05d.png'

    do_eval(args.data_dir, args.dataset, args.crfs, FRAME_NAME)    