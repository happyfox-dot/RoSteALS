from  kornia.augmentation import RandomGaussianBlur, RandomGaussianNoise, RandomMedianBlur, RandomBrightness, RandomSaturation, RandomContrast,RandomHue

import torchvision 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

class tResize(nn.Module):
    def __init__(self, resize_ratio, interpolation_method = 'nearest'):
        super().__init__()
        self.resize_ratio = resize_ratio
        self.interpolation_method = interpolation_method
        self.trans = torchvision.transforms.Resize(256)
    def forward(self, input_image):
        img = F.interpolate(input_image, scale_factor= (self.resize_ratio,self.resize_ratio), mode = self.interpolation_method)
        return self.trans(img)
    

class GaussainNoisy(nn.Module):
    def __init__(self, mean, std ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.aug  = RandomGaussianNoise(mean = self.mean,std = self.std, p = 1.)
    
    def forward(self, input_image):
        return self.aug(input_image)
        

class GaussainBluer(nn.Module):
    def __init__(self, kernel_size ,sigma ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.aug = RandomGaussianBlur(kernel_size = (self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma), p = 1.)
    
    def forward(self, input_image):
        return self.aug(input_image)    

class median_blur(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.aug = RandomMedianBlur(kernel_size=self.kernel_size, p = 1.)

    def forward(self, input_image):
        return self.aug(input_image)
    

# class salt_and_pepper(nn.Module):
#     def __init__(self, ratio = 0.1):
#         super().__init__()
#         self.ratio = ratio 
#         self.aug = SaltAndPepper(ratio=self.ratio )
#     def forward(self, input_image):
#         return self.aug(input_image)
    
class random_brightness(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.aug = RandomBrightness(brightness=(self.a, self.b ), p = 1.)
    
    def forward(self, input_image):
        input_image = self.aug(input_image)
        return input_image
    
class random_contrast(nn.Module):
    def __init__(self,a ,b):
        super().__init__()
        self.a = a
        self.b = b
        self.aug = RandomContrast(contrast=(self.a, self.b ), p = 1.)
    def forward(self, input_image):
        input_image = self.aug(input_image)
        return input_image
    
class random_saturation(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.aug = RandomSaturation(saturation= (self.a, self.b ), p= 1.)

    def forward(self, input_image):
        input_image = self.aug(input_image)
        return input_image
    
class random_hue(nn.Module):
    def __init__(self,a ,b):
        super().__init__()
        self.a = a 
        self.b = b
        self.aug = RandomHue(hue =(self.a,self.b ), p = 1.)

    def forward(self, input_image):
        input_image = self.aug(input_image)
        return input_image

        




def main():
    # img = torch.rand(1,1,32, 32)
    # rescale = Resize(0.5)
    # output_img = rescale(img)
    # print(output_img.shape)
    # # gau_noisy= GaussainNoisy(0, 0.1)
    # # output = gau_noisy(img)
    # # print(img)
    # # print(output)   
    # # gau_blur = GaussainBluer(7, 0.1)
    # # out_blur = gau_blur(img)
    # # print(out_blur) 

    pass



if __name__ == '__main__':
    main()


