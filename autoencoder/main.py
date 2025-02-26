import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from absl import app
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim

from src_autoencoder.config import FLAGS
from src_autoencoder.load_model import load_model

from src_autoencoder.tarf_dataloader import TarfDataloader
from src_autoencoder.validation import validation
from model.encoder import *
from model.decoder import *

import wandb
from datetime import datetime

class psnr_loss(nn.Module):
    def __init__(self):
        super(psnr_loss, self).__init__()
        
    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return -(20 * torch.log10(1.0 / torch.sqrt(mse)))


def get_current_datetime():
    # Get the current date and time
    current_datetime = datetime.now()
    # Format it as needed, e.g., "YYYY-MM-DD HH:MM:SS"
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    return formatted_datetime

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def save_image(image, dir, filename):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, filename)
    
    for i in range(image.size(0)):
        img = rescale(image[i], (-1, 1), (0, 255), clamp=True)
        
        img = img.detach().to('cpu')
        img_pil = reverse_transforms(img)
        img_pil.save(f'{filename}_{i}.png')

def main(argv):
    # 0. Set random seed
    torch.manual_seed(FLAGS.seed)
    starting_time = get_current_datetime()
    
    # 0.1 Initialize wandb
    if FLAGS.wandb:
        wandb.init(project=FLAGS.wandb_project, group=FLAGS.wandb_group, name=starting_time, config=FLAGS)
        wandb.config.update(FLAGS)
    
    # 1. Load data
    train_dataset = TarfDataloader(FLAGS, train=True)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
    test_dataset = TarfDataloader(FLAGS, train=False)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)
    
    # 2. Define model
    enc = VAE_Encoder().to(FLAGS.device)
    dec = VAE_Decoder().to(FLAGS.device)
    
    if FLAGS.pretrained:
        enc, dec = load_model(enc, dec, device=FLAGS.device)
    
    ssim_ls = ssim(data_range=1.)
    l2 = nn.MSELoss()
    l1 = nn.L1Loss()
    psnr_ls = psnr_loss()
    # rec_loss = vae_loss_function()
    
    # 3. Define optimizer
    optimizer_enc = optim.Adam(enc.parameters(), lr=FLAGS.lr, weight_decay=1e-9)
    optimizer_dec = optim.Adam(dec.parameters(), lr=FLAGS.lr, weight_decay=1e-9)
    #scheduler_enc = optim.lr_scheduler.MultiStepLR(optimizer_enc, milestones=[12, 20, 26], gamma=0.1)
    #scheduler_dec = optim.lr_scheduler.MultiStepLR(optimizer_dec, milestones=[12, 20, 26], gamma=0.1)
    
    # 4. Train model
    # set tqdm
    total_epoch_loss = 0
    for e in range(FLAGS.epochs):
        with tqdm(train_loader, desc=f'Epoch {e}', unit="Batch") as tepoch:
            for i, data in enumerate(tepoch):
                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
                
                data = data.to(FLAGS.device)
                
                # latent_shape = (data.size(0), 4, data.size(2)//8, data.size(3)//8)
                noise_shape = torch.ones(data.size(0), 4, data.size(2)//8, data.size(3)//8, device=FLAGS.device)
                noise = torch.randn_like(noise_shape, device=FLAGS.device)
                
                z = enc(data, noise=noise)
                output = dec(z)
                
                loss = l2(data, output)

                loss.backward()
                optimizer_enc.step()
                optimizer_dec.step()
                
                total_epoch_loss += loss.item()
                
                tepoch.set_postfix(loss=total_epoch_loss/(i+1))
                
                if e % 5 == 0 or e == FLAGS.epochs-1:
                    # print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')
                    if FLAGS.log:
                        if FLAGS.pretrained:
                            save_image(data, os.path.join("autoencoder_touch_results/pretrained", starting_time), 'original')
                            save_image(output, os.path.join("autoencoder_touch_results/pretrained", starting_time), 'reconstructed')
                        else:
                            save_image(data, os.path.join("autoencoder_touch_results", starting_time), 'original')
                            save_image(output, os.path.join("autoencoder_touch_results", starting_time), 'reconstructed')
                
        # TODO: implement validation step and pass val_loss to scheduler
        if e % 5 == 0 or e == FLAGS.epochs-1:
            val_loss = validation(enc, dec, l2, test_loader, starting_time, FLAGS)
            print(f"Validation loss: {-val_loss}")
        
        
        #scheduler_enc.step(val_loss)
        #scheduler_dec.step(val_loss)
        #print(f"LR: {scheduler_enc.get_last_lr()}")
        
        total_epoch_loss /= len(train_loader)
        # wandb log
        if FLAGS.wandb:
            wandb.log({"loss": total_epoch_loss, "val_loss": val_loss, "epoch": e, "lr": scheduler_enc.get_last_lr()[0]})
    
        if e % 5 == 0 or e == FLAGS.epochs-1:
            # save weights in path
            if FLAGS.log:
                print(f"Saving epoch {e} weights...")
                if FLAGS.pretrained:
                    weights_path = 'weights/touch/autoencoder/pretrained'
                else:
                    weights_path = 'weights/touch/autoencoder'
                if not os.path.exists(weights_path):
                    os.makedirs(weights_path)
                torch.save(enc.state_dict(), f'{weights_path}/encoder_{starting_time}.pth')
                torch.save(dec.state_dict(), f'{weights_path}/decoder_{starting_time}.pth')
    
if __name__ == '__main__':
    app.run(main)