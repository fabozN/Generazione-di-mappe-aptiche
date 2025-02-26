import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

class TarfDataloader(Dataset):
    def __init__(self, flags, train=True):
        self.data_dir = flags.data_dir
        self.train = train
        
        self.breaking_point = flags.breaking_point
        self.test_breaking_point = flags.test_breaking_point
        
        data_path = os.path.join(self.data_dir, 'touch_results')
        
        self.transform = transforms.Compose([
            transforms.Resize((flags.image_size, flags.image_size)),
            transforms.ToTensor()
        ])


        self.data = []
        files = os.listdir(data_path)
        for cnt, file in enumerate(files):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.data.append(os.path.join(data_path, file))
            
            if self.breaking_point > 0 and cnt == self.breaking_point:
                break
        
#        dirs = os.listdir(data_path)
#        train_dirs = dirs[:20]
#        test_dirs = dirs[20:]
        
#        self.train_data = []
#        for d in train_dirs:
#            current_path = os.path.join(data_path, d)
#            files = os.listdir(current_path)
#            for cnt, file in enumerate(files):
#                if file.endswith('.jpg'):
#                    self.train_data.append(os.path.join(current_path, file))
#                
#                if self.breaking_point > 0:
#                    if cnt == self.breaking_point:
#                        break
#            if self.breaking_point > 0:
#                if cnt == self.breaking_point:
#                    break
                    
#        self.test_data = []
#        for d in test_dirs:
#            current_path = os.path.join(data_path, d)
#            files = os.listdir(current_path)
#            for file in files:
#                if file.endswith('.jpg'):
#                    self.test_data.append(os.path.join(current_path, file))
#                
#                if self.test_breaking_point > 0:
#                    if cnt == self.breaking_point:
#                        break
#            if self.test_breaking_point > 0:
#                if cnt == self.breaking_point:
#                    break
        
        if self.train:
            print("Training data loaded!!")
        else:    
            print("Testing data loaded!!")
    
            
#    def __len__(self):
#        return self.len_train() if self.train else self.len_test()
    
#    def len_train(self):
#        return len(self.train_data)
    
#    def len_test(self):
#        return len(self.test_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
#       if self.train:
        img_path = self.data[index]
        img = Image.open(img_path)
        img = self.transform(img)[:3]
        img = rescale(img, (0, 1), (-1, 1))
#       else:
#            img_path = self.test_data[index]
#            img = Image.open(img_path)
#            img = self.transform(img)[:3]
#            img = rescale(img, (0, 1), (-1, 1))
        
        return img
    
    