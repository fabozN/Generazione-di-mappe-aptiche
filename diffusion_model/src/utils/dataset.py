import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
#import cv2
import torch

from src.utils.utils import rescale


class TaRF_RGB_NOCS_DataLoader(Dataset):
    def __init__(self, flags, train=False):
        self.data_dir = flags.data_dir
        self.train = train
        self.split_train = flags.split_train
        self.cond_options = flags.cond_options
        
        self.breaking_point = flags.breaking_point
        self.test_breaking_point = flags.test_breaking_point
        
        touch_data_path = os.path.join(self.data_dir, 'touch_results')
        vision_data_path = os.path.join(self.data_dir, 'vision_results')
        nocs_data_path = os.path.join(self.data_dir, 'nocs_results')
        
        self.transform = transforms.Compose([
            transforms.Resize((flags.image_size, flags.image_size)),
            transforms.ToTensor()
        ])
        
        touch_items = sorted(os.listdir(touch_data_path))
        vision_items = sorted(os.listdir(vision_data_path))
        nocs_items = sorted(os.listdir(nocs_data_path))
        
        self.item_list = []
        cnt = 0
        #for counter, (touch, vision, depth, vision_small, depth_small, nocs) in enumerate(zip(touch_items, vision_items, nocs_items)):
        for counter, (touch, vision, nocs) in enumerate(zip(touch_items, vision_items, nocs_items)):    
            # check if name corresponds:
            #assert touch.split('.')[0] == vision.split('.')[0] == nocs.split('.')[0] == vision_small.split('.')[0], "Names do not correspond"
            assert touch.split('.')[0] == vision.split('.')[0] == nocs.split('.')[0], "Names do not correspond"

            cnt += 1
            self.item_list.append((os.path.join(touch_data_path, touch),
                              os.path.join(vision_data_path, vision),
                              os.path.join(nocs_data_path, nocs)))
            
            if self.breaking_point > 0:
                if cnt == self.breaking_point:
                    break
        
        # split item_list into chunks of 50 elements
        chunks = [self.item_list[i:i + 50] for i in range(0, len(self.item_list), 50)]
        
        train_prop = 0.8
        val_prop = 0.1
        test_prop = 0.1
        self.train_list = []
        self.val_list = []
        self.test_list = []
        for chunk in chunks:
            self.train_list.append(chunk[:int(len(chunk) * train_prop)])
            self.val_list.append(chunk[int(len(chunk) * train_prop):int(len(chunk) * train_prop)+int(len(chunk) * val_prop)])
            self.test_list.append(chunk[int(len(chunk) * train_prop)+int(len(chunk) * val_prop):])
        
        # transform list of list in one list only
        self.train_list = [item for sublist in self.train_list for item in sublist]
        self.val_list = [item for sublist in self.val_list for item in sublist]
        self.test_list = [item for sublist in self.test_list for item in sublist]
        
        self.conditioning_shape = 0
        if 'rgb' in self.cond_options:
            self.conditioning_shape += 3
        if 'nocs' in self.cond_options:
            self.conditioning_shape += 4
        
        
        if train:
            print(f"Loaded train data: {self.len_train()} samples")
        else:
            print(f"Loaded test data: {self.len_test()} samples")
        
    def __len__(self):
        return self.len_train() if self.train else self.len_test()
    
    def len_train(self):
        return len(self.train_list)
    
    def len_test(self):
        return len(self.test_list)
    
    def get_conditioning_shape(self):
        return self.conditioning_shape
    
    def __getitem__(self, index):
        if self.train:
            touch, vision, nocs = self.train_list[index]
        else:
            touch, vision, nocs  = self.test_list[index]
            
        touch = Image.open(touch)
        touch = np.array(touch).astype(np.uint8)
        crop = min(touch.shape[0], touch.shape[1])
        h, w, = touch.shape[0], touch.shape[1]
        touch = touch[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        touch = Image.fromarray(touch)
        touch = self.transform(touch)
        touch = rescale(touch, (0, 1), (-1, 1))
        
        vision = Image.open(vision)
        vision = np.array(vision).astype(np.uint8)
        crop = min(vision.shape[0], vision.shape[1])
        h, w, = vision.shape[0], vision.shape[1]
        vision = vision[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        vision = Image.fromarray(vision)
        vision = self.transform(vision)
        
        nocs = Image.open(nocs)
        nocs = np.array(nocs).astype(np.uint8)
        crop = min(nocs.shape[0], nocs.shape[1])
        h, w, = nocs.shape[0], nocs.shape[1]
        nocs = nocs[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        nocs = Image.fromarray(nocs)
        nocs = self.transform(nocs)
        
        input = touch
        conditioning = torch.zeros(0)
        if 'rgb' in self.cond_options:
            conditioning = torch.cat([conditioning, vision], 0)
        if 'nocs' in self.cond_options:
            conditioning = torch.cat([conditioning, nocs], 0)
                
            # conditioning = torch.cat([vision, depth, vision_small, depth_small, nocs, bg], 0)
        # else:
            # print(f"Testing data not implemented yet!!")
            # input = None
            # conditioning = None
        
        assert input is not None, "Input is None"
        assert conditioning is not None, "Conditioning is None"
        
        return input, conditioning
        
        
        
def main(argv):
    data = TaRF_RGB_NOCS_DataLoader(FLAGS, train=True)
    train = DataLoader(data, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    item, cond = train.dataset.__getitem__(0)
    print("Finished debugging!!")
        
if __name__ == '__main__':
    from absl import app
    from config.config import FLAGS
    app.run(main)