import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from options import CDTrainOptions
import functools
# from einops import rearrange
# import torchvision.models as models
import models
# from models.CG import SegNetEnc
# from models.help_funcs2 import Transformer, TransformerDecoder, TwoLayerConv2d
# from FCD_Module import *


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net=net.cuda()
        # net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class EncoderAD(nn.Module):
    def __init__(self):
        super(EncoderAD, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.enc1 = nn.Sequential(resnet18.conv1, resnet18.bn1, resnet18.relu)
        self.enc2 = nn.Sequential(resnet18.layer1)
        self.enc3 = nn.Sequential(resnet18.layer2)
        self.enc4 = nn.Sequential(resnet18.layer3)
        self.enc5 = nn.Sequential(resnet18.layer4)

        self.AD5 = SegNetEnc(512, 256, 2, 1)
        self.AD4 = SegNetEnc(256, 128, 2, 1)
        self.AD3 = SegNetEnc(128, 64, 2, 1)
        self.AD2 = SegNetEnc(64, 32, 2, 1)
        self.AD1 = SegNetEnc(64, 32, 2, 1)
        self.apply(self.init_model)

    def forward(self, x, y):
        x_f1 = self.enc1(x)
        x_f2 = self.enc2(x_f1)
        x_f3 = self.enc3(x_f2)
        x_f4 = self.enc4(x_f3)
        x_f5 = self.enc5(x_f4)

        y_f1 = self.enc1(y)
        y_f2 = self.enc2(y_f1)
        y_f3 = self.enc3(y_f2)
        y_f4 = self.enc4(y_f3)
        y_f5 = self.enc5(y_f4)

        ad_f5 = self.AD5(abs(x_f5 - y_f5))
        ad_f4 = self.AD4(abs(x_f4 - y_f4))
        ad_f3 = self.AD3(abs(x_f3 - y_f3))
        ad_f2 = self.AD2(abs(x_f2 - y_f2))
        ad_f1 = self.AD1(abs(x_f1 - y_f1))

        return ad_f1, ad_f2, ad_f3, ad_f4, ad_f5

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True
class DecoderAD(nn.Module):
    def __init__(self):
        super(DecoderAD, self).__init__()

        self.GC5 = SegNetEnc(256,256,1,1)
        self.GC4 = SegNetEnc(384,128,1,1)
        self.GC3 = SegNetEnc(192,64,1,1)
        self.GC2 = SegNetEnc(96,32,1,1)
        self.GC1 = SegNetEnc(64,32,1,1)
        self.predection = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        # self.apply(self.init_model)
    def forward(self, ad1, ad2, ad3, ad4, ad5):

        # 512 256 128 64 32
        ad5 = self.GC5(ad5)
        ad4 = self.GC4(torch.cat([F.upsample_bilinear(ad5, scale_factor=2), ad4], 1))
        ad3 = self.GC3(torch.cat([F.upsample_bilinear(ad4, scale_factor=2), ad3], 1))
        ad2 = self.GC2(torch.cat([F.upsample_bilinear(ad3, scale_factor=2), ad2], 1))
        ad1 = self.GC1(torch.cat([F.upsample_bilinear(ad2, scale_factor=1), ad1], 1))
        out = self.predection(ad1)
        return ad1, out
class WSCD(nn.Module):
    def __init__(self):
        super(WSCD, self).__init__()
        self.encoder  = EncoderAD()
        self.decoder1 = DecoderAD()
        self.decoder2 = DecoderAD()
    def forward(self, C, UC, test=False):
        ad1, ad2, ad3, ad4, ad5 = self.encoder(C[0], C[1])
        feat1, cmask1 = self.decoder1(ad1, ad2, ad3, ad4, ad5)
        feat2, cmask2 = self.decoder2(ad1, ad2, ad3, ad4, ad5)
        if test:
            return feat1,cmask1
        uad1, uad2, uad3, uad4, uad5 = self.encoder(UC[0], UC[1])
        ufeat1, ucmask1 = self.decoder1(uad1, uad2, uad3, uad4, uad5)
        ufeat2,ucmask2 = self.decoder2(uad1, uad2, uad3, uad4, uad5)
        sc_cd1 = ad1 + uad1
        sc_cd2 = ad2 + uad2
        sc_cd3 = ad3 + uad3
        sc_cd4 = ad4 + uad4
        sc_cd5 = ad5 + uad5
        scf1, scmask1 = self.decoder1(sc_cd1,sc_cd2,sc_cd3,sc_cd4,sc_cd5)
        scf2, scmask2 = self.decoder2(sc_cd1,sc_cd2,sc_cd3,sc_cd4,sc_cd5)
        return feat1, cmask1,feat2, cmask2,ufeat1, ucmask1,ufeat2,ucmask2,scmask1,scmask2



