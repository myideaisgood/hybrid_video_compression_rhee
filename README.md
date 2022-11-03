# video_compression_cvpr2023_rhee

Implementation of "Lightweight Hybrid Video Compression Framework Using Reference-Guided Restoration Network"

## Environments
- Ubuntu 18.04
- Pytorch 1.7.0
- CUDA 10.0.130 & cuDNN 7.6.5
- Python 3.7.7

You can type the following command to easily build the environment.
Download 'fdnet_env.yml' and type the following command.

```
conda env create -f fdnet_env.yml
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


## Guideline for Data Processing

Extract frames from original yuv/y4m files

```
ffmpeg -i FILE.y4m FILE/f%05d.png

```

Frames to HEVC video
```
ffmpeg -i FILE/f%05d.png -c:v hevc -preset medium -x265-params bframes=0 -crf %d .mp4

```

HEVC video to frames
```
ffmpeg -i .mp4 /f%05d.png

```
