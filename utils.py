import io
import torchvision.transforms as transforms
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2 
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image 
import PIL
from os import getcwd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import math

PATH = getcwd()
augs_transforms = transforms.Compose([transforms.ToTensor()])

# VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# class CNNColorizer(nn.Module):
#     def __init__(self, in_channels = 1):
#         super(CNNColorizer, self).__init__()
#         self.in_channels = in_channels
#         self.encoder = self.create_conv_layers(VGG16)
#         self.linear1 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512*7*7, 64)
#         )
#         self.linear2 = nn.Sequential(
#             nn.Linear(64, 512*7*7),
#             nn.ReLU()
#         )
#         self.decoder= nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size = (3,3), stride= (1,1), padding=(1,1), bias = False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Upsample(scale_factor = 2),
#             nn.ConvTranspose2d(256, 128, kernel_size = (3,3), stride= (1,1), padding=(1,1), bias = False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Upsample(scale_factor = 2),
#             nn.ConvTranspose2d(128, 64, kernel_size = (3,3), stride= (1,1), padding=(1,1), bias = False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Upsample(scale_factor = 2),
#             nn.ConvTranspose2d(64, 32, kernel_size = (3,3), stride= (1,1), padding=(1,1), bias = False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Upsample(scale_factor = 2),
#             nn.Conv2d(32, 2, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
#             nn.Tanh(),
#             nn.Upsample(scale_factor = 2)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

#     def create_conv_layers(self, architecture):
#         layers =  []
#         in_channels = self.in_channels

#         for x in architecture:
#             if type(x) == int:
#                 out_channels = x
#                 layers += [
#                     nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
#                     nn.BatchNorm2d(x),
#                     nn.Tanh()
#                     ]
#                 in_channels = x
#             elif x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]

#         return nn.Sequential(*layers)

class CNNColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU()
        )
        
        self.decoder= nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose2d(32, 16, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose2d(16, 8, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(8, 2, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.Tanh(),
            nn.Upsample(scale_factor = 2)
        )
        
    def forward (self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    
def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel

def Upsample(in_channels, out_channels, scale):
    layers = []
    for i in range(int(math.log(scale, 2))):
        layers += [nn.Conv2d(in_channels, out_channels*4, 3, 1, 1),
                   nn.PixelShuffle(2),
                   nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                   nn.ReLU()]
        if i == 0:
            layers[-2].weight.data.copy_(icnr(layers[-2].weight.data))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.fin = nn.Sequential(
            nn.Conv2d(channels*4, channels, 1, 1)
        )

    def forward(self, x):
        res = x
        x1 = self.conv(x) + x
        x2 = self.conv(x1) + x1 + x
        x3 = self.conv(x2)
        x4 = self.fin(torch.cat([x3, x2, x1, x], dim = 1))
        
        return x4 + x
    
def Upsample(in_channels, out_channels, scale):
    layers = []
    for i in range(int(math.log(scale, 2))):
        layers += [nn.Conv2d(in_channels, out_channels*4, 3, 1, 1),nn.ReLU(),nn.PixelShuffle(2)]
    return nn.Sequential(*layers)

class SuperR(nn.Module):
    def __init__(self, in_channels=1, feats=64):
        super().__init__()
        self.initS = nn.Sequential(
            nn.Conv2d(in_channels, feats, 9, 1, 4),
            nn.ReLU()
        )
        
#         self.res_blocks = nn.Sequential(*[ResBlock(feats) for _ in range(num_res_blocks)])
        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(64)
        self.res_block3 = ResBlock(64)
        self.res_block4 = ResBlock(64)
        self.res_block5 = ResBlock(64)
        self.res_block6 = ResBlock(64)
        
        self.inter = nn.Sequential(
            nn.Conv2d(feats*6, feats, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feats, feats, 3, 1, 1)
        )
        
        self.upsample = Upsample(64, 64, 2)
        
        self.fin = nn.Sequential(
            nn.Conv2d(feats, in_channels, 9, 1, 4),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        d1 = self.initS(x)
        d2 = self.res_block1(d1)
        d3 = self.res_block2(d2)
        d4 = self.res_block3(d3)
        d5 = self.res_block3(d4)
        d6 = self.res_block3(d5)
        d7 = self.res_block3(d6)
        d8 = self.inter(torch.cat([d2, d3, d4, d5, d6, d7], dim = 1))
        d9 = self.upsample(d8 + d1)
        return self.fin(d9)

device = torch.device('cpu')
model1 = CNNColorizer().to(device=device)
model2 = SuperR().to(device = device)

# MODEL_PATH1 = PATH + '/model/VGGNet-10eps-batchnorm-image-colorization-data-kaggle-20-12-22.t7'
MODEL_PATH1 = PATH + '/model/CNNColorizer-50eps-flickr-data-kaggle.t7'

checkpoint1 = torch.load(MODEL_PATH1, map_location='cpu')
model1.load_state_dict(checkpoint1['state_dict'])

MODEL_PATH2 = PATH + '/model/SuperR-JH-0_19302498251199723-eps.t7'

checkpoint2 = torch.load(MODEL_PATH2, map_location='cpu')
model2.load_state_dict(checkpoint2['state_dict'])



def trans_img(input_image):
    img = cv2.imread(PATH + '/'+input_image)
    og = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = resize(img, (512,512))
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    lab = rgb2lab(img)
    X = lab[:, :, 0]
    X = X.reshape(X.shape+(1,))
    og = cv2.cvtColor(og, cv2.COLOR_BGR2RGB).astype(np.float32)
    og = resize(og, (512,512))
    # cv2.imwrite(f"{PATH}/static/temp/input.png", og)
    return augs_transforms(X).unsqueeze(0), og
#y = lab[:, :, 1:].transpose((2, 0, 1))
# y = lab[:, :, 1:]
# y = (y/128).astype(np.float32)


def colorize(imagename):
    l, og = trans_img(input_image = imagename)
    l = l.to(device = device)
    op = model1(l)
    op = op*128
    op = op[0].permute(1,2,0).detach().numpy()

    result = np.zeros((512,512,3))
    result[:, :, 0:1] = l.cpu()[0].permute(1,2,0).numpy()
    result[:, :, 1:2 ] = op[:,:,0:1]
    result[:, :, 2:3 ] = op[:,:,1:2]
    colorimage = lab2rgb(result)
    colorimage = 255*(colorimage - np.min(colorimage))/(np.max(colorimage) - np.min(colorimage))
    colorimage = colorimage.astype(np.uint8)
    colorimage = cv2.cvtColor(colorimage, cv2.COLOR_BGR2RGB)

    # cv2.imwrite(f"{PATH}/static/temp/output.png", colorimage)
    _, input_img = cv2.imencode('.jpg', og)  
    input_bytes = input_img.tobytes()
    input_b64 = base64.b64encode(input_bytes)
    input_b64 = input_b64.decode('utf-8')

    _, output_img = cv2.imencode('.jpg', colorimage)  
    output_bytes = output_img.tobytes()
    output_b64 = base64.b64encode(output_bytes)
    output_b64 = output_b64.decode('utf-8')

    result = {
        "input":f'data:image/jpg;base64,{input_b64}',
        "output":f'data:image/jpg;base64,{output_b64}'
    }

    return result

def trans_img2(input_image):
    # img = cv2.imread(PATH + '/' + input_image)
    img = Image.open(PATH + '/' + input_image)
    og = img

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    return augs_transforms(img_y).unsqueeze(0), img_cb, img_cr, og

def superres(imagename):
    img_y, img_cb, img_cr, og = trans_img2(imagename)
    img_y = img_y.to(device = device)
    model2.eval()
    op = model2(img_y)
    op = op.cpu()
    op = op[0].detach().numpy()

    img_out_y = Image.fromarray(np.uint8((op * 255.0).clip(0, 255)[0]), mode='L')
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")
    
    buff1 = BytesIO()

    final_img.save(buff1, format="JPEG")
    output_bytes = buff1.getvalue()
    output_b64 = base64.b64encode(output_bytes)
    output_b64 = output_b64.decode("utf-8")

    buff2 = BytesIO()
    og = og.resize(final_img.size, Image.BICUBIC)
    og.save(buff2, format="JPEG")
    input_bytes = buff2.getvalue()
    input_b64 = base64.b64encode(input_bytes)
    input_b64 = input_b64.decode("utf-8")

    result = {
        "input":f'data:image/jpg;base64,{input_b64}',
        "output":f'data:image/jpg;base64,{output_b64}'
    }

    return result



# imsave('colored.jpg', lab2rgb(result))

# colorize('bandw.jpg')

# plt.imshow(colorimage)
# plt.show()