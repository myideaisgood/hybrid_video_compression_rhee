import os
import logging

def do_conventional(data_dir, dataset, crfs, FRAME_NAME):


    HEVC_DIR = 'hevc_result/'
    
    # Create ../DATASET/hevc_result/dataset/
    # Example : ../DATASET/hevc_result/UVG/
    os.makedirs(os.path.join(data_dir, HEVC_DIR))
    os.makedirs(os.path.join(data_dir, HEVC_DIR, dataset))

    DATA_DIR = os.path.join(data_dir, dataset)

    logging.basicConfig(filename='log_conventional.txt',format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    logging.info('*** Start ***')

    video_names = sorted([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])

    for crf in crfs:

        # Create ../DATASET/hevc_result/dataset/crf/
        os.makedirs(os.path.join(data_dir, HEVC_DIR, dataset, str(crf)))

        SAVE_DIR = os.path.join(data_dir, HEVC_DIR, dataset, str(crf))

        # Do HEVC if needed
        for video_name in video_names:
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


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='UNet')
    parser.add_argument('--crf', type=int, default=22, help='crf')
    parser.add_argument('--data_dir', type=str, default='../DATASET/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='UVG/', help='Dataset name')

    args = parser.parse_args()


    if 'vimeo' in args.dataset:
        FRAME_NAME = 'im%d.png'
    else:
        FRAME_NAME = 'f%05d.png'

    do_conventional(args.data_dir, args.dataset, [args.crf], FRAME_NAME)