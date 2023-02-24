# Lightweight Hybrid Video Compression Framework Using Reference-Guided Restoration Network (VVC Version)

Implementation of "Lightweight Hybrid Video Compression Framework Using Reference-Guided Restoration Network" (VVC Version)


## Dataset

[Vimeo90k] (http://toflow.csail.mit.edu/)

[UVG] (https://ultravideo.fi/#testsequences)

[MCL-JCV] (http://mcl.usc.edu/mcl-jcv-dataset/)

[HEVC Class B] (https://github.com/HEVC-Projects/CPH/blob/master/README.md)


# Installation of VVC (VVenC/VVdeC)

[VVenC] (https://github.com/fraunhoferhhi/vvenc)
[VVdeC] (https://github.com/fraunhoferhhi/vvdec)

## Brief explanation of contents

```
|── experiments
    ├──> experiment_name 
         └──> ckpt : trained models will be saved here        
|── utils/data_loaders.py : data pipeline
|── utils/helpers.py : utiliy functions
|── config.py : configuration should be controlled only here 
|── eval_conventional.py : evaluates the conventional video codec (VVC) for configured dataset
|── eval.py : evaluate the model
|── model.py : architecture of proposed method
|── process_data.py : performs VVC to raw video data
└── train.py : train the model

```


## Command line for VVC processing

1. Compress the original YUV videos into VVC compressed files
VVC setting is 'medium'
```
vvencapp --preset medium -i VIDEO.y4m -s 1920x1080 --qp QP --qpa 1 -r 24 -o OUTPUT.266
```

2. Extract frames from the VVC compressed video
```
vvdecapp --bitstream OUTPUT.266 --output OUTPUT.y4m
ffmpeg -i OUTPUT.y4m VIDEO/f%05d.png
```


## Using the VVC commands, your dataset should look something like this

Compressed frames should look like

```
|── VVC
    ├──> UVG
        ├──> Video1
            ├──> Video1_QP1.266, Video1_QP1.y4m, Video1_QP2.266, Video1_QP2.y4m, ...
            ├──> QP1
                └──> f00001.png, f00002.png, ... , f00600.png
            ├──> QP2
                └──> f00001.png, f00002.png, ... , f00600.png                
            ...
            └──> video2
        ├──> Video2
        ├──> Video3
        └──> Video4
    └──> MCL-JCV
        ├──> Video1
            ├──> Video1_QP1.266, Video1_QP1.y4m, Video1_QP2.266, Video1_QP2.y4m, ...
            ├──> QP1
                └──> f00001.png, f00002.png, ... , f00600.png
            ├──> QP2
                └──> f00001.png, f00002.png, ... , f00600.png                
            ...
            └──> video2
        ├──> Video2
        ├──> Video3
        └──> Video4
```

## Guidelines for Codes

1. Run the following command for to train the network step1.

```
python train.py --gpu_num=0 --qp=21 --train_step='step1' --exp_name='default/' --train_dataset='vimeo/'
```

2. Run the following command for to train the network step2.

```
python train.py --gpu_num=0 --qp=21 --train_step='step2' --exp_name='default/' --train_dataset='vimeo/'
```

3. Run the following command for to evaluate the network.

```
python eval.py --gpu_num=0 --qp=21 --train_step='step2' --exp_name='default/' --eval_dataset='UVG/'
```
