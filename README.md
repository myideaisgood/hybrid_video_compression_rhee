# Lightweight Hybrid Video Compression Framework Using Reference-Guided Restoration Network

Implementation of "Lightweight Hybrid Video Compression Framework Using Reference-Guided Restoration Network"

## Environments
- Ubuntu 18.04
- Pytorch 1.11.0
- CUDA 10.2 & cuDNN 7.6.5
- Python 3.7.13

You can type the following command to easily build the environment.
Download 'compression.yml' and type the following command.

```
conda env create -f compression.yml
```

## Abstract

Recent deep-learning-based video compression methods brought coding gains over conventional codecs such as AVC and HEVC. However, learning-based codecs generally require considerable computation time and model complexity. In this paper, we propose a new lightweight hybrid video codec consisting of a conventional video codec, a lossless image codec, and our new restoration network. Precisely, our encoder consists of the HEVC video encoder and a lossless image encoder, transmitting a lossy-compressed video bitstream along with a losslessly-compressed reference frame. The decoder is constructed with corresponding video/image decoders and a new restoration network, which enhances the compressed video in two-step processes. In the first step, a network trained with a large video dataset restores the details lost by the HEVC encoder. Then, we further boost the video quality with the guidance of a reference image, which is a losslessly compressed video frame. The reference image provides video-specific information, which can be utilized to better restore the details of a compressed video. Experimental results show that the proposed method achieves comparable performance to top-tier methods with lower complexity and faster run time.

## Dataset

[Vimeo90k] (http://toflow.csail.mit.edu/)

[UVG] (https://ultravideo.fi/#testsequences)

[MCL-JCV] (http://mcl.usc.edu/mcl-jcv-dataset/)

[HEVC Class B] (https://github.com/HEVC-Projects/CPH/blob/master/README.md)


## Brief explanation of contents

```
|── experiments
    ├──> experiment_name 
         └──> ckpt : trained models will be saved here        
|── utils/data_loaders.py : data pipeline
|── utils/helpers.py : utiliy functions
|── config.py : configuration should be controlled only here 
|── eval_conventional.py : evaluates the conventional video codec (HEVC) for configured dataset
|── eval.py : evaluate the model
|── model.py : architecture of proposed method
|── process_data.py : performs HEVC to raw video data
└── train.py : train the model

```


## Command line for HEVC processing

1. Extract uncompressed frames from original yuv,y4m files

```
ffmpeg -i video.y4m video/f%05d.png
```

2. Convert uncompressed frames to HEVC compressed video (quantization factor of CRF)
   HEVC setting is 'medium'
```
ffmpeg -i video/f%05d.png -c:v hevc -preset medium -x265-params bframes=0 -crf CRF video.mp4
```

3. Extract frames from HEVC compressed video
```
ffmpeg -i video.mp4 FILE/f%05d.png
```


## Using the HEVC commands, your dataset should look something like this

Uncompressed frames should look like

```
|── UVG
    ├──> video1
        └──> f00001.png, f00002.png, ... , f00600.png
    ├──> video2
    ...
    └──> video7
└── MCL-JCV
    ├──> video1
        └──> f00001.png, f00002.png, ... , f00600.png
    ├──> video2
    ...
    └──> video15
```


Compressed frames should look like

```
|── hevc_result
    ├──> UVG
        ├──> 21
            ├──> video1_21.mp4, video2_21.mp4, ... video7_21.mp4
            ├──> video1
                └──> f00001.png, f00002.png, ... , f00600.png
            ├──> video2
                └──> f00001.png, f00002.png, ... , f00600.png                
            ...
            └──> video2
        ├──> 23
        ├──> 25
        └──> 27
    └──> MCL-JCV
        ├──> 21
            ├──> video1_21.mp4, video2_21.mp4, ... video7_21.mp4
            ├──> video1
                └──> f00001.png, f00002.png, ... , f00600.png
            ├──> video2
                └──> f00001.png, f00002.png, ... , f00600.png                
            ...
            └──> video2
        ├──> 23
        ├──> 25
        └──> 27
```

## Guidelines for Codes

1. Run the following command for to train the network step1.

```
python train.py --gpu_num=0 --crf=21 --train_step='step1' --exp_name='default/' --train_dataset='vimeo/'
```

2. Run the following command for to train the network step2.

```
python train.py --gpu_num=0 --crf=21 --train_step='step2' --exp_name='default/' --train_dataset='vimeo/'
```

3. Run the following command for to evaluate the network.

```
python eval.py --gpu_num=0 --crf=21 --train_step='step2' --exp_name='default/' --eval_dataset='UVG/'
```
