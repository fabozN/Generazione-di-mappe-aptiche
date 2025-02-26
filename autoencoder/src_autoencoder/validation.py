import os
import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

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
        # transforms.Lambda(lambda t: (t + 1) / 2),
        # transforms.Lambda(lambda t: t * 255.),
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

def validation(enc, dec, loss_fn, dataloader, starting_time, FLAGS):
    enc.eval()
    dec.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(dataloader, desc='Validation', unit='Batch') as t:
            for i, batch in enumerate(t):
                x = batch.to(FLAGS.device)
                noise_shape = torch.ones(x.size(0), 4, x.size(2)//8, x.size(3)//8, device=FLAGS.device)
                noise = torch.randn_like(noise_shape, device=FLAGS.device)
                z = enc(x, noise=noise)
                x_hat = dec(z)
                loss = loss_fn(x, x_hat)
                total_loss += loss.item()
                
                if FLAGS.log:
                    if FLAGS.pretrained:
                        save_image(x, os.path.join("autoencoder_touch_results/pretrained", starting_time), 'original')
                        save_image(x_hat, os.path.join("autoencoder_touch_results/pretrained", starting_time), 'reconstructed')
                    else:
                        save_image(x, os.path.join("autoencoder_touch_results", starting_time), 'original')
                        save_image(x_hat, os.path.join("autoencoder_touch_results", starting_time), 'reconstructed')
                    
                    
    return total_loss / len(dataloader)