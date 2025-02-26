import os
os.chdir("/home/fabo/codiceWSL/autoencoder/autoencoder")
from absl import app
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from model.encoder import VAE_Encoder
from model.decoder import VAE_Decoder
from torch.utils.data import DataLoader, Dataset
from src_autoencoder.tarf_dataloader import TarfDataloader
from tqdm import tqdm

from test_psnr_config import FLAGS

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return (20 * torch.log10(1.0 / torch.sqrt(mse)))

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
        # img = image[i]
        
        img = img.detach().to('cpu')
        img_pil = reverse_transforms(img)
        img_pil.save(f'{filename}_{i}.png')


def main(argv):
    # 1. Define the model
    enc = VAE_Encoder().to(FLAGS.device)
    dec = VAE_Decoder().to(FLAGS.device)

    # 2. Load the model
    #encoder_pth = "weights/touch/autoencoder/encoder_2024-10-13-40-54.pth"
    encoder_pth = "/home/fabo/codiceWSL/autoencoder/autoencoder/weights/touch/autoencoder/encoder_2025-02-19-11-30.pth"
    encoder_name = os.path.splitext(os.path.basename(encoder_pth))[0]
    #decoder_pth = "weights/touch/autoencoder/decoder_2024-10-13-40-54.pth"
    decoder_pth = "/home/fabo/codiceWSL/autoencoder/autoencoder/weights/touch/autoencoder/decoder_2025-02-19-11-30.pth"
    decoder_name = os.path.splitext(os.path.basename(decoder_pth))[0]

    enc.load_state_dict(torch.load(encoder_pth, weights_only=False))
    dec.load_state_dict(torch.load(decoder_pth, weights_only=False))

    # 3. Load the data
    train_dataset = TarfDataloader(FLAGS, train=True)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=8)
    test_dataset = TarfDataloader(FLAGS, train=False)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=8)

    # 4. Compute the PSNR
    enc.eval()
    dec.eval()
    
    dir = f"psnr_results/{encoder_name}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    average_psnr_for_all = 0
    with torch.no_grad():
        with tqdm(train_loader, unit="Batch") as tepoch:
            for idx, data in enumerate(tepoch):
                average_psnr_for_batch = 0
                data = data.to(FLAGS.device)
                
                noise_shape = torch.ones(data.size(0), 4, data.size(2)//8, data.size(3)//8, device=FLAGS.device)
                noise = torch.randn_like(noise_shape, device=FLAGS.device)
                z = enc(data, noise=noise)
                output = dec(z)
                
                # Save the PSNR in file for each image in batch
                for i in range(data.size(0)):
                    psnr = compute_psnr(data[i], output[i])
                    
                    average_psnr_for_batch += psnr.item()
                    average_psnr_for_all += psnr.item()
                    with open(f"psnr_results/{encoder_name}/psnr_train_samples.txt", "a") as f:
                        f.write(f"PSNR: {psnr.item()}\n")
                
                print(f"Average PSNR for batch: {average_psnr_for_batch/data.size(0)}")
                
                # Save some sample images each 10 batches
                if idx % 100 == 0:
                    # rescale(data, (-1, 1), (0, 255), clamp=True)
                    # rescale(output, (-1, 1), (0, 255), clamp=True)
                    save_image(data, f"{dir}/train_samples/batch_{idx}", f"original")
                    save_image(output, f"{dir}/train_samples/batch_{idx}", f"reconstructed")
                
                
                
    print(f"Average PSNR for all test: {average_psnr_for_all/len(train_loader.dataset)}")
    with open(f"{dir}/psnr_train_samples.txt", "a") as f:
        f.write(f"\nAverage PSNR for all: {average_psnr_for_all/len(train_loader.dataset)}")
        
    average_psnr_for_all = 0
    with torch.no_grad():
        with tqdm(test_loader, unit="Batch") as tepoch:
            for idx, data in enumerate(tepoch):
                average_psnr_for_batch = 0
                data = data.to(FLAGS.device)
                
                noise_shape = torch.ones(data.size(0), 4, data.size(2)//8, data.size(3)//8, device=FLAGS.device)
                noise = torch.randn_like(noise_shape, device=FLAGS.device)
                z = enc(data, noise=noise)
                output = dec(z)
                
                # Save the PSNR in file for each image in batch
                for i in range(data.size(0)):
                    psnr = compute_psnr(data[i], output[i])
                    
                    average_psnr_for_batch += psnr.item()
                    average_psnr_for_all += psnr.item()
                    with open(f"{dir}/psnr_test_samples.txt", "a") as f:
                        f.write(f"PSNR: {psnr.item()}\n")
                
                print(f"Average PSNR for batch: {average_psnr_for_batch/data.size(0)}")
                
                # Save some sample images each 10 batches
                if idx % 100 == 0:
                    # rescale(data, (-1, 1), (0, 255), clamp=True)
                    # rescale(output, (-1, 1), (0, 255), clamp=True)
                    save_image(data, f"{dir}/test_samples/batch_{idx}", f"original")
                    save_image(output, f"{dir}/test_samples/batch_{idx}", f"reconstructed")
                
    print(f"Average PSNR for all test: {average_psnr_for_all/len(test_loader.dataset)}")
    with open(f"{dir}/psnr_test_samples.txt", "a") as f:
        f.write(f"\nAverage PSNR for all: {average_psnr_for_all/len(test_loader.dataset)}")


if __name__ == '__main__':
    app.run(main)