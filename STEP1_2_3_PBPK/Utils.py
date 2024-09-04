import datetime
import torch
import scipy.io as sio
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
#from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import scipy.misc
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import math
from random import choice, sample, seed, randint, random, gauss
import gc
import pandas as pd
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
import os

class PBPKLoss(nn.Module):
  def __init__(self,opt):
    super(PBPKLoss, self).__init__()
    self.opt = opt
    self.criterion = nn.MSELoss()
  
  def contrast_agent_concentration_measured(self, si, si_t):
    T1 = 5.3 #0.820 T1 = 5.08 #0.820
    r1 = 2.6 #4.5 r1 = 2.39 #4.5
    RE = torch.divide(torch.subtract( si_t, si.view(si_t.shape[0], -1) ), si.view(si_t.shape[0], -1))
    Ct_measured = torch.divide(RE, T1*r1)
    return Ct_measured
 
  
  def forward(self, output_mask, roi, curve, weight=1):
    w = torch.cuda.FloatTensor(weight).fill_(weight)
    if self.opt.cuda:
      w.cuda()
    w = Variable(w, requires_grad=False)
    #calcolo la media per ogni tempo
    #output_mask è una slice tutta nera ad eccezione della zona della lesione
    #quindi nel calcolare la media devo escludere il contorno della lesione
    #torch.sum(output_mask, axis = (2,3)) somma tutti i pixel della lesione per ogni canale quindi abbiamo batchx3
    # torch.sum(roi,axis=(2,3)) ci dice il numero di pixel per canale
    sum_1 = torch.sum(roi,axis=(2,3))
    if sum_1.any() == 0:
      print(f'ERROR PBPK LOSS: {sum_1}')
    media_pixel = torch.divide( torch.sum(output_mask, axis = (2,3)), torch.sum(roi,axis=(2,3)) )  #batchx3.. abbiamo 3 pixel per ogni slice

    ct_output = self.contrast_agent_concentration_measured(media_pixel[:,0], media_pixel) #-> 40,3
    loss=w*self.criterion(ct_output, curve)
    return loss

    
class BiasReduceLoss(nn.Module):
    def __init__(self,opt):
        super(BiasReduceLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()
    def forward(self, x, y, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.avg = torch.mean(x,0).unsqueeze(0)
        self.loss = w*self.criterion(self.avg, y)
        return self.loss

class TotalVaryLoss(nn.Module):
    def __init__(self,opt):
        super(TotalVaryLoss, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + 
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss

class SelfSmoothLoss2(nn.Module):
    def __init__(self,opt):
        super(SelfSmoothLoss2, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.loss = torch.sum(torch.mul(self.x_diff, self.x_diff)) + torch.sum(torch.mul(self.y_diff, self.y_diff))
        self.loss = w * self.loss
        return self.loss 


###################################
#########  basic blocks  ##########
###################################
# a mixer (linear layer)
class waspMixer(nn.Module):
    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspMixer, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# shading * albedo = texture(img)
class waspIntrinsicComposer(nn.Module):
    def __init__(self, opt):
        super(waspIntrinsicComposer, self).__init__()
        self.ngpu = opt.ngpu
        self.nc = opt.nc
    def forward(self, shading, albedo):
        self.shading = shading.repeat(1,self.nc,1,1)
        self.img = torch.mul(self.shading, albedo)
        return self.img

# warp image according to the grid
class waspWarper(nn.Module):
    def __init__(self, opt):
        super(waspWarper, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize = opt.imgSize

    def forward(self, input_img, input_grid):
        self.warp = input_grid.permute(0,2,3,1)
        self.output = F.grid_sample(input_img, self.warp, align_corners=True )
        return self.output

# integrate over the predicted grid offset to get the grid(deformation field)
class waspGridSpatialIntegral(nn.Module):
    def __init__(self,opt):
        super(waspGridSpatialIntegral, self).__init__()
        self.opt = opt
        self.w = self.opt.imgSize
        self.filterx = torch.cuda.FloatTensor(1,1,1,self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if opt.cuda:
            self.filterx.cuda()
            self.filtery.cuda()
    def forward(self, input_diffgrid):
        #print(input_diffgrid.size())
        fullx = F.conv_transpose2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w], fully[:,:,0:self.w,0:self.w]),1)
        return output_grid

# an encoder architecture
class waspEncoder(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output   

# a decoder architecture
class waspDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderTanh(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            #nn.Hardtanh(lb,ub),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output




###################################
###### encoders and decoders ######        
###################################

#### The encoders ####

# encoders of DAE
class Encoders(nn.Module):
    def __init__(self, opt):
        super(Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# encoders of instrinsic DAE
class Encoders_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        #self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        #self.zImg  = self.zImixer(self.z)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp

# encoders of DAE, using DenseNet architecture
class Dense_Encoders(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# encoders of Intrinsic DAE, using DenseNet architecture
class Dense_Encoders_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp

#### The decoders ####

# decoders of DAE
class DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.warper   = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


# decoders of intrinsic DAE  -------> modificato da Michela
#DecodersIntegralWarper2_Intrinsic
class DecodersIntegralWarper2_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_Intrinsic, self).__init__()

        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.nc   = opt.nc     #mi prendo i canali di output

        self.decoderS = waspDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderT = waspDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper   = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)

    def forward(self, zS, zT, zW, basegrid, roi):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1)) #shading
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1)) #albedo
        self.img = self.intrinsicComposer(self.shading, self.texture) #texture

        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid

        self.output  = self.warper(self.img, self.warping) #applico a img il campo di deformazione per ottenere l'immagine complessiva

        self.warpedAlbedo = self.warper(self.texture, self.warping) #applico il campo solo all'albedo, ottenendo un warped albedo
        roi = roi.repeat(1,self.nc,1,1)  #replico la roi per averla a 3 canali
        self.masked_out =  torch.mul(roi, self.output)  #maschero l'output

        #DA CONSIDERARE SOLO PER LO STEP DELLE LESIONI INVENTATE
        #applico la deformazione anche alla roi della slice corrispondente all'albedo
        self.newroi = self.warper(roi, self.warping) 
        #devo binanizzare la roi
        self.newroi = (self.newroi > 0).type(torch.cuda.FloatTensor)
        #a questo punto creo la masked out che mi serve per la loss 
        self.masked_out_fake = torch.mul(self.newroi, self.output)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping , self.masked_out, self.warpedAlbedo, self.newroi, self.masked_out_fake


# decoders of DAE, using DenseNet architecture 
class Dense_DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.warper   = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.img = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.img, self.resWarping, self.output, self.warping

# decoders of Intrinsic DAE, using DenseNet architecture
"""class Dense_DecodersIntegralWarper2_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        
        # shading decoder
        self.decoderS = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        # albedo decoder
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        # deformation decoder
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        # shading*albedo=texture
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        # deformation offset decoder
        self.warper   = waspWarper(opt)
        # spatial intergrator for deformation field
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img     = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping"""

class Dense_DecodersIntegralWarper2_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.nc   = opt.nc     #mi prendo i canali di output
        
        # shading decoder
        self.decoderS = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        # albedo decoder
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        # deformation decoder
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        # shading*albedo=texture
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        # deformation offset decoder
        self.warper   = waspWarper(opt)
        # spatial intergrator for deformation field
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)

    def forward(self, zS, zT, zW, basegrid, roi):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img     = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)

        #prevedo una uscita in più-> maschero l'output. Ovvero considero solo la zona dell'output dove dovrebbe esserci la lesione
        self.warpedAlbedo = self.warper(self.texture, self.warping) #applico il campo solo all'albedo, ottenendo un warped albedo
        roi = roi.repeat(1,self.nc,1,1)  #replico la roi
        self.masked_out =  torch.mul(roi, self.output)  #maschero l'output

        #DA CONSIDERARE SOLO PER LO STEP DELLE LESIONI INVENTATE
        #applico la deformazione anche alla roi della slice corrispondente all'albedo
        self.newroi = self.warper(roi, self.warping) 
        #devo binanizzare la roi
        self.newroi = (self.newroi > 0).type(torch.cuda.FloatTensor)
        #a questo punto creo la masked out che mi serve per la loss 
        self.masked_out_fake = torch.mul(self.newroi, self.output)
        
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping , self.masked_out, self.warpedAlbedo, self.newroi, self.masked_out_fake



###################################
########  densenet blocks #########
###################################
class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no 
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no 
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        return self.main(inputs)

class waspDenseEncoder(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Sigmoid, f_args=[]):
        super(waspDenseEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim

        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.BatchNorm2d(nc),
                nn.ReLU(True),
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

                # state size. (ndf) x 32 x 32
                DenseBlockEncoder(ndf, 6),
                DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation=activation, args=args),

                # state size. (ndf*2) x 16 x 16
                DenseBlockEncoder(ndf*2, 12),
                DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation=activation, args=args),

                # state size. (ndf*4) x 8 x 8
                DenseBlockEncoder(ndf*4, 24),
                DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation=activation, args=args),

                # state size. (ndf*8) x 4 x 4
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation=activation, args=args),
                f_activation(*f_args),
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        return output   

class waspDenseDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh, f_args=[0,1]):
        super(waspDenseDecoder, self).__init__()
        self.ngpu   = ngpu
        self.main   = nn.Sequential(
            # input is Z, going into convolution
            nn.BatchNorm2d(nz),
            activation(*args),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*4),

            # state size. (ngf*4) x 8 x 8
            DenseBlockDecoder(ngf*4, 24),
            DenseTransitionBlockDecoder(ngf*4, ngf*2),

            # state size. (ngf*2) x 16 x 16
            DenseBlockDecoder(ngf*2, 12),
            DenseTransitionBlockDecoder(ngf*2, ngf),

            # state size. (ngf) x 32 x 32
            DenseBlockDecoder(ngf, 6),
            DenseTransitionBlockDecoder(ngf, ngf),

            # state size (ngf) x 64 x 64
            nn.BatchNorm2d(ngf),
            activation(*args),
            nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )
    def forward(self, inputs):
        return self.main(inputs)


def Contrast_Agent_Concentration_misured(si, si_t):
  T1 = 5.3 #0.820 T1 = 5.08 #0.820
  r1 = 2.6 #4.5 r1 = 2.39 #4.5
  #si è un vettore di 9 elementi -> escludiamo il primo
  RE = np.divide(np.subtract(si_t, si), si)
  Ct_misured = np.divide(RE, T1*r1)
  return Ct_misured


def Weinmann_AIF(time):
  Cp=np.zeros(10)
  D = 0.1
  a1 = 3.99
  a2 = 4.78
  m1 = 0.144
  m2 = 0.0111
  n_time = time.shape[0]
  for i in range(0,n_time):
    Cp[i] = D * ((a1 * math.exp(-m1*time[i])) + (a2 * math.exp(-m2*time[i])))

  return Cp

def Parker_AIF(time):
  A1 = 0.809
  A2 = 0.330
  T1 = 0.17046
  T2 = 0.365
  sigma1 = 0.0563
  sigma2 = 0.132
  alpha = 1.050
  beta = 0.1685
  s = 38.078
  tau = 0.483

  Cp=np.zeros(10)
  pdf_1 = scipy.stats.norm(loc = T1, scale = sigma1)
  pdf_2 = scipy.stats.norm(loc = T2, scale = sigma2)
  n_time = time.shape[0]

  for i in range(0,n_time):
    Cp[i] = A1*pdf_1.pdf(time[i]) + A2*pdf_2.pdf(time[i]) + alpha*((math.exp(-beta*time[i]))/(1 + math.exp(-s*(time[i] - tau))))
  return Cp

#------------------------------------------------------------------------- AIF SINGLE
def Weinmann_AIF_single(time):
  D = 0.1
  a1 = 3.99
  a2 = 4.78
  m1 = 0.144
  m2 = 0.0111
  Cp = D * ((a1 * math.exp(-m1*time)) + (a2 * math.exp(-m2*time)))
  return Cp

def Parker_AIF_single(time):
  A1 = 0.809
  A2 = 0.330
  T1 = 0.17046
  T2 = 0.365
  sigma1 = 0.0563
  sigma2 = 0.132
  alpha = 1.050
  beta = 0.1685
  s = 38.078
  tau = 0.483

  pdf_1 = scipy.stats.norm(loc = T1, scale = sigma1)
  pdf_2 = scipy.stats.norm(loc = T2, scale = sigma2)

  Cp = A1*pdf_1.pdf(time) + A2*pdf_2.pdf(time) + alpha*((math.exp(-beta*time))/(1 + math.exp(-s*(time - tau))))
  return Cp


def integral_trap(t,y): 
  return np.sum(np.multiply( t[1:]-t[:-1], (y[:-1] + y[1:])/2. ) )

def TK_model_Weinmann_integral_trap(time,ktrans, ve):
  Ct = np.zeros(10)
  Cp = Weinmann_AIF(time)
  n_time = time.shape[0]
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y) 
  return Ct

def ETK_model_Weinmann_integral_trap(time,ktrans, ve, vp):
  Ct = np.zeros(10)
  n_time = time.shape[0]
  Cp = Weinmann_AIF(time)
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y)  + vp*Cp[i]
  return Ct

def TK_model_Parker_integral_trap(time,ktrans, ve):
  Ct = np.zeros(10)
  Cp = Parker_AIF(time)
  n_time = time.shape[0]
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y) 
  return Ct

def ETK_model_Parker_integral_trap(time,ktrans, ve, vp):
  Ct = np.zeros(10)
  Cp = Parker_AIF(time)
  n_time = time.shape[0]
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y)  + vp*Cp[i]
  return Ct


def setCuda(*args):
    barg = []
    for arg in args: 
        barg.append(arg.cuda())
    return barg

def setAsVariable(*args):
    barg = []
    for arg in args: 
        barg.append(Variable(arg))
    return barg
                        
class BalanceConcatDataset(ConcatDataset):  
  def __init__(self, datasets):
    l = max([len(dataset) for dataset in datasets])
    new_l = l
    
    for dataset in datasets: 
      old_samples = dataset.samples #in questo modo estraggo i campioni dal set iniziale
      while len(dataset) < new_l:
        #sample is without replacement
        dataset.samples += sample(old_samples, min(len(dataset), new_l - len(dataset))) #change 
    super(BalanceConcatDataset, self).__init__(datasets)


def getBaseGrid(N=64, normalize = True, getbatch = False, batchSize = 1):
    a = torch.arange(-(N-1), (N), 2)
    if normalize:
        a = a/(N-1.0)
    x = a.repeat(N,1)
    y = x.t()
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)),0)
    if getbatch:
        grid = grid.unsqueeze(0).repeat(batchSize,1,1,1)
    return grid
    