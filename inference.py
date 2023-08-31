
import os, sys, torch 
import argparse
from pathlib import Path
import numpy as np
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid
import lpips
from tools.sifid import SIFID
from tools.helpers import welcome_message
from tools.ecc import ECC
import torch 
from tqdm import trange
import random
from noisy.Jpeg import Jpeg 
from noisy.salt_and_papper import SaltAndPepper
from noisy.noisy import GaussainBluer, GaussainNoisy, tResize, random_brightness,random_hue, random_contrast,random_saturation,median_blur

# def unormalize(x):
#     # convert x in range [-1, 1], (B,C,H,W), tensor to [0, 255], uint8, numpy, (B,H,W,C)
#     x = torch.clamp((x + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#     return x
photo_path = r'/mnt/chengxin/Datasets/DUTS/DUTS-TE/Std-Image-30/'

random.seed(42)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(welcome_message())
    # Load model
    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    model = model.cuda()
    model.eval()

    # load images nums
    source_images = os.listdir(photo_path)
    source_images = sorted(source_images)
    source_images = source_images[len(source_images) //2:]
    print(f'source_image_length:{len(source_images)}')

    # cover
    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
     # secret
    ecc = ECC()
    secret = ecc.encode_text([args.secret])  # 1, 100

    secret = torch.from_numpy(secret).cuda().float()  # 1, 100
    # 
    # jp = Jpeg(factor=90)
    # for quality in ([10, 30 , 50, 70, 90]):
        # jp  = Jpeg(factor=quality)
    # for sig in ([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]): # kernel_default= 7 
    # for sig in ([0.01, 0.02, 0.03, 0.04, 0.05]):
    # for sig in ([0.5, 0.75, 0.9, 1., 1.25, 1.5, 2.0]):
    # for sig in ([0.9]):
    name = [median_blur(kernel_size= 5), SaltAndPepper(ratio=0.1), random_brightness(a=0.8, b=1.2), 
            random_contrast(a = 0.8 , b= 1.2), random_saturation(a = 0.8, b=1.2), random_hue(a = -0.1, b = 0.1)]
    for noi in range(6):
        donoisy = name[noi]
        bar= []
        ssim = []
        psnr = []
        # br = GaussainBluer(kernel_size= 7 , sigma= sig )
        # br = GaussainNoisy(mean = 0 , std = sig)
        # br = tResize(resize_ratio= sig)
        with trange(len(source_images)) as t :
            for i , source_image in zip(t, source_images):
                # cover_org is the origianal image 
                cover_org = Image.open(os.path.join(photo_path, source_image)).convert('RGB')
                # get photo size 

                # w,h = cover_org.size
                w, h = cover_org.size 
                # transforms
                # cover size is 
                cover = tform(cover_org).unsqueeze(0).cuda()  # 1, 3, 256, 256

            
                # inference
                # lpips_alex = lpips.LPIPS(net='alex').cuda()
                # sifid_model = SIFID()
                with torch.no_grad():
                    z = model.encode_first_stage(cover)
                    z_embed, _ = model(z, None, secret)
                    stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256

                    res = stego.clamp(-1,1) - cover  # (1,3,256,256) residual
                    # resize this image size , 
                    res = torch.nn.functional.interpolate(res, (h,w), mode='bilinear')
                    res = res.permute(0,2,3,1).cpu().numpy()  # (1,h,w,3)
                    stego_uint8 = np.clip(res[0] + np.array(cover_org)/127.5-1., -1,1)*127.5+127.5  
                    stego_uint8 = stego_uint8.astype(np.uint8)  # (h,w, 3), ndarray, uint8

                    # quality metrics
                    # print(f'Quality metrics at resolution: {h}x{w} (HxW)')

                    # print(f'MSE: {compute_mse(np.array(cover_org)[None,...], stego_uint8[None,...])}')
                    psnr.append(compute_psnr(np.array(cover_org)[None,...], stego_uint8[None,...]))
                    ssim.append(compute_ssim(np.array(cover_org)[None,...], stego_uint8[None,...]))
                    # print(f'PSNR: {compute_psnr(np.array(cover)[None,...], stego_uint8[None,...])}')
                    # print(f'SSIM: {compute_ssim(np.array(cover )[None,...], stego_uint8[None,...])}')
                    
                    # cover_org_norm = torch.from_numpy(np.array(cover_org)[None,...]/127.5-1.).permute(0,3,1,2).float().cuda()
                    # stego_norm = torch.from_numpy(stego_uint8[None,...]/127.5-1.).permute(0,3,1,2).float().cuda()
                    # calculate  lpips and  sifid metric
                    # print(f'LPIPS: {compute_lpips(cover_org_norm, stego_norm, lpips_alex)}')
                    # print(f'SIFID: {compute_sifid(cover_org_norm, stego_norm, sifid_model)}')

                    # decode secret
                    # print('Extracting secret...')
                    # 
                    # print(f'setgo image shape is :{stego.shape}')
                    
                    # break
                    # stego [batch, channels, height, width] , [1,3, 256, 256]
                    # TODO:: noisy 

                    # random 


                    stego = donoisy(stego)
                    # stego = jp(stego.cpu())
                    # stego = br(stego)






                    secret_pred = (model.decoder(stego) > 0).cpu().numpy()  # 1, 100
                    # print(secret_pred)
                    # print(f'secret is : {secret}')
                    # break
                    # print(np.mean(secret_pred == secret.cpu().numpy()))
                    bar.append(np.mean(secret_pred == secret.cpu().numpy()))
                    # if i == 20:
                    #     break
                    # print(f'Bit acc: {np.mean(secret_pred == secret.cpu().numpy())}')

                    # using ecc correction to get the 
                    # secret_decoded = ecc.decode_text(secret_pred)[0]
                    # print(f'Recovered secret: {secret_decoded}')

                    # output_filename= r'watermarking_image/'  + source_image
                    # # save stego
                    # Image.fromarray(stego_uint8).save(output_filename, quality = 100)
                    # break
                    # print(f'Stego saved to {source_image}')
        # print(f'sigma is :{sig}')
        print(f'Jpeg , quality = {donoisy}')
        print(f'mean psnr is :{np.mean(psnr)}')

        print(f'mean ssim is :{np.mean(ssim)}')
        # print(bar)
        print(f'meam bar is :{np.mean(bar):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # load config file
    parser.add_argument('-c', "--config", default='models/VQ4_s100_mir100k2.yaml', help="Path to config file.")
    parser.add_argument('-w', "--weight", default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/VQ4_s100_mir100k2/checkpoints/epoch=000017-step=000449999.ckpt', help="Path to checkpoint file.")
    # resize photot  to 
    parser.add_argument(
        "--image_size", type=int, default=256, help="Height and width of square images."
    )
    parser.add_argument(
        "--secret", default='secrets', help="secret message, 7 characters max"
    )
    # # wait to embed image photo
    # parser.add_argument(
    #     "--cover", default='examples/00096.png', help="cover image path"
    # )
    # # output file name
    # parser.add_argument(
    #     "-o", "--output", default='stego.png', help="output stego image path"
    # )
    args = parser.parse_args()
    main(args)