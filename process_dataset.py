import os
import cv2
import numpy as np
import math
import logging

def do_vvc(data_dir, dataset, qp):

    SAVE_DIR = 'VVC/'

    logging.basicConfig(filename='log_vvc.txt',format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    logging.info('*** Start ***')

    video_names = sorted(os.listdir(os.path.join(data_dir, dataset + '_raw')))

    os.makedirs(os.path.join(data_dir, SAVE_DIR), exist_ok=True)
    os.makedirs(os.path.join(data_dir, SAVE_DIR, dataset), exist_ok=True)
        
    for video_name in video_names:
        IN_VIDEO = os.path.join(data_dir, dataset + '_raw', video_name)
            
        vid_name = video_name.split('.')[0]

        os.makedirs(os.path.join(data_dir, SAVE_DIR, dataset, vid_name), exist_ok=True)

        OUT_VIDEO = os.path.join(data_dir, SAVE_DIR, dataset, vid_name, vid_name + '_' + str(qp) + '.266')
        DEC_VIDEO = os.path.join(data_dir, SAVE_DIR, dataset, vid_name, vid_name + '_' + str(qp) + '.y4m')

        if not os.path.exists(OUT_VIDEO):
            os.system('vvencapp --preset medium -i %s -s 1920x1080 --qp %d --qpa 1 -r 24 -o %s' % (IN_VIDEO, qp, OUT_VIDEO))
            os.system('vvdecapp --bitstream %s --output %s' % (OUT_VIDEO, DEC_VIDEO))

        DEC_FRAMES = os.path.join(data_dir, SAVE_DIR, dataset, vid_name, str(qp))

        if not os.path.exists(DEC_FRAMES):
            os.makedirs(DEC_FRAMES, exist_ok=True)
            os.system('ffmpeg -i %s %s' % (DEC_VIDEO, os.path.join(DEC_FRAMES, "f%05d.png")))

    logging.info('*** End ***')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--qp', type=int, default=22, help='qp')
    parser.add_argument('--data_dir', type=str, default='../DATASET/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='UVG/', help='Dataset name')

    args = parser.parse_args()

    do_vvc(args.data_dir, args.dataset, [args.qp])