## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm
import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx
import lpips

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/motion_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str,
                    help='Test Dataset')  # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--original_dir', default='./originals/', type=str)
args = parser.parse_args()

####### Load yaml #######
# yaml_file = '/opt/restormer/Motion_Deblurring/Options/Deblurring_Restormer.yml'
yaml_file = '/opt/restormer/Defocus_Deblurring/Options/DefocusDeblur_Single_8bit_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

factor = 8
dataset = args.dataset
# result_dir  = os.path.join(args.result_dir, dataset)
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

# inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
inp_dir = args.input_dir
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))

lpips_loss_fn = lpips.LPIPS(net='alex')  # You can choose a different network architecture if needed
lpips_loss_fn.cuda()
psnr_values = []
lpips_values = []
# Set up logging
log_file_path = os.path.join(args.result_dir, 'celeba_gauss_deblurring_0p05_log.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)
logging.info(f'{args.weights} with Noise 0.05 for CelebA Dataset')

with torch.no_grad():
    # for file_ in tqdm(files):
    for idx, file_ in enumerate(tqdm(files)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_)) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                       img_as_ubyte(restored))

        # Load original image
        orig_img = np.float32(utils.load_img(args.original_dir + f"/orig_{idx}.png")) / 255.
        orig_img = torch.from_numpy(orig_img).permute(2, 0, 1)
        orig_img = orig_img.unsqueeze(0).cuda()
        # Calculate PSNR
        mse = torch.mean((orig_img.cpu() - torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0)).to('cuda') ** 2)
        psnr = 10 * torch.log10(1 / mse)
        psnr_values.append(psnr.item())

        # Calculate LPIPS
        lpips_loss = lpips_loss_fn(orig_img, torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0).cuda())
        lpips_values.append(lpips_loss.item())

        # Log values for each iteration
        logging.info(f'Image {idx}, PSNR: {psnr.item():.4f} dB, LPIPS: {lpips_loss.item():.4f}')

# Calculate and log averages
average_psnr = np.mean(psnr_values)
average_lpips = np.mean(lpips_values)

logging.info(f'Average PSNR: {average_psnr:.4f} dB')
logging.info(f'Average LPIPS: {average_lpips:.4f}')

# Close the logging file handler
logging.shutdown()
