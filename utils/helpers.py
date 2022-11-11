import math
import numpy as np
import cv2

import torch
import torch.nn as nn

def var_or_cuda(x, device=None):
    x = x.contiguous()
    if torch.cuda.is_available() and device != torch.device('cpu'):
        if device is None:
            x = x.cuda(non_blocking=True)
        else:
            x = x.cuda(device=device, non_blocking=True)

    return x

def img2cuda(imgs, device):

    for idx, img in enumerate(imgs):
        imgs[idx] = var_or_cuda(img, device=device)
    
    return imgs

def get_psnr_np(original, compressed):
    original = original.astype(np.float)
    compressed = compressed.astype(np.float)

    mse = np.mean((original - compressed)**2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def get_psnr(output, image, PIXEL_MAX=1):

    mse = (output - image)**2
    mse = torch.mean(mse)
    psnr = 10 * torch.log10(PIXEL_MAX**2/mse)

    return psnr

def tensor2np(tensor_img):

    np_img = tensor_img[0].permute(1,2,0).detach().cpu().numpy()
    np_img = (255*np_img).astype(np.uint8)

    return np_img

def tensor2img(tensor_img):

    np_img = tensor_img.permute(1,2,0).numpy()
    np_img = (255*np_img).astype(np.uint8)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    return np_img

def get_diff_img(img1, img2):

    diff = abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff = np.mean(diff, axis=2)
    diff = diff.astype(np.uint8)
    diff = np.stack([diff, diff, diff], axis=-1)

    return diff

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)      

def count_module_parameters(model):

    trainable_list = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            names = name.split('.')
            if names[0] not in trainable_list:
                trainable_list.append(names[0])

    module_params = []

    for module_param in trainable_list:
        module_param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and module_param in name:
                module_param_count += param.numel()

        module_params.append(module_param_count)

    total_params = sum(module_params)

    return trainable_list, module_params, total_params    