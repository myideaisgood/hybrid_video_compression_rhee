import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from config import parse_args
from utils.helpers import *

def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer

class EnhancerBlock(nn.Module):
    def __init__(self, in_channel, out_channel, act):
        super(EnhancerBlock, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.act = ActivationLayer(act)
    
    def forward(self, x):
        x = self.conv(x)
        out = self.act(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = ActivationLayer(act)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out += residual
        out = self.act(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, act):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.act = ActivationLayer(act)
    
    def forward(self, x):
        x = self.conv(x)
        return self.act(x)    

class step1_encoder(nn.Module):
    def __init__(self, args):
        super(step1_encoder, self).__init__()

        layer_num = len(args.step1_channels)

        channels = [3] + args.step1_channels

        self.layers = nn.ModuleList()

        for idx in range(layer_num):
            self.layers.append(nn.Conv2d(channels[idx], channels[idx+1], 3, 1, 1))
            self.layers.append(ActivationLayer(args.activation_type))
            self.layers.append(ResBlock(channels[idx+1], channels[idx+1], args.activation_type))    

    def forward(self, c_t):

        f = c_t

        for layer in self.layers:
            f = layer(f)

        return f

class step1_decoder(nn.Module):
    def __init__(self, args):
        super(step1_decoder, self).__init__()

        channels = args.step1_channels[-1]

        self.head_layer = nn.Conv2d(channels, 3, 1, 1)
        
        self.img_out_type = args.img_out_type

    def forward(self, f_step1, c_t):

        res = self.head_layer(f_step1)

        img_out = res + c_t

        if self.img_out_type == 'none':
            img_out = img_out
        elif self.img_out_type == 'tanh':
            img_out = (torch.tanh(img_out) + 1) * 0.5
        elif self.img_out_type == 'sigmoid':
            img_out = torch.sigmoid(img_out)        

        return img_out

class step2_encoder(nn.Module):
    def __init__(self, args):
        super(step2_encoder, self).__init__()

        layer_num = len(args.step2_enc_channels)

        channels = [3] + args.step2_enc_channels

        self.layers = nn.ModuleList()

        for idx in range(layer_num):
            self.layers.append(nn.Conv2d(channels[idx], channels[idx+1], 3, 1, 1))
            self.layers.append(ActivationLayer(args.activation_type))
            self.layers.append(ResBlock(channels[idx+1], channels[idx+1], args.activation_type))    


    def forward(self, x_ref):

        f = x_ref

        for layer in self.layers:
            f = layer(f)

        return f

class step2_decoder(nn.Module):
    def __init__(self, args):
        super(step2_decoder, self).__init__()

        layer_num = len(args.step2_dec_channels)

        channels = [args.step1_channels[-1]] + args.step2_dec_channels

        self.layers = nn.ModuleList()

        for idx in range(layer_num):
            self.layers.append(nn.Conv2d(channels[idx], channels[idx+1], 3, 1, 1))
            self.layers.append(ActivationLayer(args.activation_type))
            self.layers.append(ResBlock(channels[idx+1], channels[idx+1], args.activation_type))    

        self.head_layer = nn.Conv2d(channels[-1], 3, 1, 1)

        self.img_out_type = args.img_out_type


    def forward(self, f_step1, f_step2, confidence_map, c_t):

        f = f_step1 + confidence_map * f_step2

        for layer in self.layers:
            f = layer(f)

        res = self.head_layer(f)

        img_out = res + c_t

        if self.img_out_type == 'none':
            img_out = img_out
        elif self.img_out_type == 'tanh':
            img_out = (torch.tanh(img_out) + 1) * 0.5
        elif self.img_out_type == 'sigmoid':
            img_out = torch.sigmoid(img_out)

        return img_out

class DeformableConv2d(nn.Module):
    def __init__(self, args):
        super(DeformableConv2d, self).__init__()

        kernel_size = 3
        stride = 1
        padding = 1

        self.kernel_size = kernel_size        

        layer_num = len(args.deform_enc_channels)
        channels = [2*args.step2_enc_channels[-1]] + args.deform_enc_channels

        self.layers = nn.ModuleList()

        for idx in range(layer_num):
            self.layers.append(EnhancerBlock(channels[idx], channels[idx+1], args.activation_type))

        self.head_layers = nn.Conv2d(channels[-1], 3*kernel_size*kernel_size, 1, 1)

        self.regular_conv = nn.Conv2d(args.step2_enc_channels[-1], args.step2_enc_channels[-1] + 1, kernel_size, stride, padding, bias=False)
    
    def forward(self, f_step1, f_step2):

        f_in = torch.cat([f_step1, f_step2], dim=1)

        for layer in self.layers:
            f_in = layer(f_in)
        om = self.head_layers(f_in)

        offset = om[:,0:2*self.kernel_size*self.kernel_size]
        mask = om[:,2*self.kernel_size*self.kernel_size:]
        mask = torch.sigmoid(mask)

        output = deform_conv2d(f_step2, offset, self.regular_conv.weight, self.regular_conv.bias, stride=1, padding=int((self.kernel_size-1)/2), mask=mask)
        
        f_deform = output[:,:-1]
        confidence_map = torch.sigmoid(output[:,-1:])

        return f_deform, confidence_map

class step2_refine(nn.Module):
    def __init__(self, args):
        super(step2_refine, self).__init__()

        layer_num = len(args.step2_dec_channels)

        channels = [args.step1_channels[-1]] + args.step2_dec_channels

        self.layers = nn.ModuleList()

        for idx in range(layer_num):
            self.layers.append(nn.Conv2d(channels[idx], channels[idx+1], 3, 1, 1))
            self.layers.append(ActivationLayer(args.activation_type))
            self.layers.append(ResBlock(channels[idx+1], channels[idx+1], args.activation_type))    

    def forward(self, f_deform):

        f = f_deform

        for layer in self.layers:
            f = layer(f)

        return f

class HochangNet(nn.Module):
    def __init__(self, args):
        super(HochangNet, self).__init__()

        self.step1_encoder = step1_encoder(args)
        self.step1_decoder = step1_decoder(args)

        self.step2_encoder = step2_encoder(args)
        self.step2_decoder = step2_decoder(args)

        self.deformable_conv = DeformableConv2d(args)
        self.step2_refine = step2_refine(args)

    def freeze_step1(self):

        for param in self.step1_encoder.parameters():
            param.requires_grad = False
        
        for param in self.step1_decoder.parameters():
            param.requires_grad = False
    
    def freeze_step2(self):
        for param in self.step2_encoder.parameters():
            param.requires_grad = False
        
        for param in self.step2_decoder.parameters():
            param.requires_grad = False        

        for param in self.deformable_conv.parameters():
            param.requires_grad = False        

        for param in self.step2_refine.parameters():
            param.requires_grad = False             

    def forward(self, c_t, x_ref):

        f_step1 = self.step1_encoder(c_t)
        pred_1 = self.step1_decoder(f_step1, c_t)

        f_step2 = self.step2_encoder(x_ref)

        # f_deform, confidence_map = self.deformable_conv(f_step1, f_step2)
        f_deform, confidence_map = self.deformable_conv(f_step2, f_step1)
        f_refine = self.step2_refine(f_deform)

        pred_2 = self.step2_decoder(f_step1, f_refine, confidence_map, c_t)

        return pred_1, pred_2

if __name__ == '__main__':

    import os

    args = parse_args()

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 1
    DOWN_SCALE = 3
    H = 1920
    W = 1080

    x_ref = torch.rand([BATCH_SIZE,3,int(H/DOWN_SCALE),int(W/DOWN_SCALE)])
    x = torch.rand([BATCH_SIZE,3,int(H/DOWN_SCALE),int(W/DOWN_SCALE)])

    x_ref = x_ref.to(device)
    x = x.to(device)

    network = HochangNet(args)
    network.to(device)

    # network.freeze_step1()
    # network.freeze_step2()

    pred_1, pred_2 = network(c_t=x, x_ref=x_ref)
    print(pred_1.shape)
    print(pred_2.shape)

    trainable_list, module_param_list, total_params = count_module_parameters(network)
    print("********** Trainable Parameters **********")
    for idx in range(len(trainable_list)):
        print("\t%15s : %.1f M" % (trainable_list[idx], module_param_list[idx] * 10**(-6)))

    print("\t%15s : %.1f M" % ('Total',total_params * 10**(-6)))