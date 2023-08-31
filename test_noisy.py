import argparse
from tqdm import trange
import numpy as np
import  os 
import torch


# 
photo_path = r'/mnt/chengxin/Datasets/DUTS/DUTS-TE/Std-Image-30/'

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    parser = argparse.ArgumentParser(description= "run test model preformance on differenet noisy")

    parser.add_argument('--noisy','-n', required=True ,type =str, help = 'noisy type' )
    args = parser.parse_args()

    print(f'noisy type is {args.noisy}')
    # load test file
    source_images = os.listdir(photo_path)
    source_images = sorted(source_images)
    source_images = source_images[len(source_images)//2:]
    print(len(source_images))

    pass 


if __name__ == '__main__':
    
    main()