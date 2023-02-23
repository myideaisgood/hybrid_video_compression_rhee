import os
import cv2
import numpy as np
import math
import logging

def get_psnr_np(original, compressed):
    original = original.astype(np.float)
    compressed = compressed.astype(np.float)

    mse = np.mean((original - compressed)**2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def do_vvc(DATA_DIR, DATASET, QPS):

    SAVE_DIR = 'VVC/'

    logging.basicConfig(filename='log_vvc.txt',format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    logging.info('*** Start ***')

    video_names = sorted(os.listdir(os.path.join(DATA_DIR, DATASET + '_raw')))

    qp_bpp_list = []
    qp_bpp_video_list = []
    qp_psnr_list = []
    qp_psnr_video_list = []

    for _ in range(len(video_names)):
        qp_bpp_video_list.append([])
        qp_psnr_video_list.append([])

    qp_idx = 0

    os.makedirs(os.path.join(DATA_DIR, SAVE_DIR), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, SAVE_DIR, DATASET), exist_ok=True)

    for qp in QPS:

        pixel_nums = 0
        video_bits = 0

        video_idx = 0
        qp_psnr = []
        
        for video_name in video_names:
            IN_VIDEO = os.path.join(DATA_DIR, DATASET + '_raw', video_name)
            
            vid_name = video_name.split('.')[0]

            os.makedirs(os.path.join(DATA_DIR, SAVE_DIR, DATASET, vid_name), exist_ok=True)

            OUT_VIDEO = os.path.join(DATA_DIR, SAVE_DIR, DATASET, vid_name, vid_name + '_' + str(qp) + '.266')
            DEC_VIDEO = os.path.join(DATA_DIR, SAVE_DIR, DATASET, vid_name, vid_name + '_' + str(qp) + '.y4m')

            if not os.path.exists(OUT_VIDEO):
                os.system('vvencapp --preset medium -i %s -s 1920x1080 --qp %d --qpa 1 -r 24 -o %s' % (IN_VIDEO, qp, OUT_VIDEO))
                os.system('vvdecapp --bitstream %s --output %s' % (OUT_VIDEO, DEC_VIDEO))

            DEC_FRAMES = os.path.join(DATA_DIR, SAVE_DIR, DATASET, vid_name, str(qp))

            if not os.path.exists(DEC_FRAMES):
                os.makedirs(DEC_FRAMES, exist_ok=True)
                os.system('ffmpeg -i %s %s' % (DEC_VIDEO, os.path.join(DEC_FRAMES, "f%05d.png")))

            # Calculate BPP
            RAW_FRAMES = os.path.join(DATA_DIR, DATASET, vid_name)
            frame_list = sorted(os.listdir(RAW_FRAMES))
            frame_num = len(frame_list)
            frame0 = cv2.imread(os.path.join(RAW_FRAMES, frame_list[0]))
            H,W,_ = frame0.shape

            vvc_frame_list = sorted(os.listdir(DEC_FRAMES))

            filesize_bit = 8*os.stat(OUT_VIDEO).st_size
            video_bits += filesize_bit
            
            pixel_nums += H*W*frame_num
            video_bpp = filesize_bit / (H*W*frame_num)

            # Calculate PSNR
            psnr_video = []
            for frame_idx in range(frame_num):
                ori_frame = cv2.imread(os.path.join(RAW_FRAMES, frame_list[frame_idx]))
                vvc_frame = cv2.imread(os.path.join(DEC_FRAMES, vvc_frame_list[frame_idx]))

                psnr_cur = get_psnr_np(ori_frame, vvc_frame)

                psnr_video.append(psnr_cur)
                qp_psnr.append(psnr_cur)
            
            video_psnr = sum(psnr_video) / len(psnr_video)

            logging.info('Video Name : %s   QP : %d   bpp : %.3f  PSNR : %.2f' % (video_name, qp, video_bpp, video_psnr))

            qp_bpp_video_list[qp_idx].append(video_bpp)
            qp_psnr_video_list[qp_idx].append(video_psnr)

        qp_bpp = video_bits / pixel_nums
        qp_psnr = sum(qp_psnr) / len(qp_psnr)

        logging.info('======= QP %d    bpp %.3f   PSNR : %.2f ======' % (qp, qp_bpp, qp_psnr))
        
        qp_bpp_list.append(qp_bpp)
        qp_psnr_list.append(qp_psnr)

        qp_idx += 1

    for idx in range(len(QPS)):
        logging.info('QP %d  BPP %.3f   PSNR %.2f' % (QPS[idx], qp_bpp_list[idx], qp_psnr_list[idx]))

    for video_idx in range(len(video_names)):
        logging.info('=== %s ===' % (video_names[video_idx]))
        for qp_idx in range(len(QPS)):
            logging.info('QP %d (%.3f, %.2f)' % (QPS[qp_idx], qp_bpp_video_list[qp_idx][video_idx], qp_psnr_video_list[qp_idx][video_idx]))

    logging.info('*** End ***')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--qps', type=int, nargs='+', default=[21, 22, 23, 24, 25, 26], help='qps')
    parser.add_argument('--data_dir', type=str, default='../DATASET/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='UVG/', help='Dataset name')

    args = parser.parse_args()

    do_vvc(args.data_dir, args.dataset, args.qps)