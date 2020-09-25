import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *
import zmf
from tqdm import tqdm

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="./logs/DnCNN-B/", help='path of log files')
parser.add_argument("--test_data", type=str, default='SetZMF', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--saveaddr", type=str, default='./saved/', help='save the results')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.jpg'))
    files_source.sort()
    # process data
    for f in tqdm(files_source):
        # image
        Img_orig = cv2.imread(f)
        Denoised_Img = np.zeros(Img_orig.shape)

        # CHANNEL 0
        Img = normalize(np.float32(Img_orig[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        INoisy = torch.Tensor(Img)
        INoisy = Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
            denoised = Out.cpu().numpy()
            Denoised_Img[:,:,2] = denoised

        # CHANNEL 1
        Img = normalize(np.float32(Img_orig[:,:,1]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        INoisy = torch.Tensor(Img)
        INoisy = Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
            denoised = Out.cpu().numpy()
            Denoised_Img[:,:,1] = denoised

        # CHANNEL 2
        Img = normalize(np.float32(Img_orig[:,:,2]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        INoisy = torch.Tensor(Img)
        INoisy = Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
            denoised = Out.cpu().numpy()
            Denoised_Img[:,:,0] = denoised

        # print(Denoised_Img.shape)
        zmf.imsave('%s/%s'%(opt.saveaddr,zmf.basename(f)),Denoised_Img)


if __name__ == "__main__":
    main()
