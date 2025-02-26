import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from tqdm import tqdm
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



def save_image(image, filename, timestep=None, old_range=None, new_range=None):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    
    for i in range(image.size(0)):
        if old_range is not None and new_range is not None:
            img = rescale(image[i], old_range, new_range, clamp=True)
        else:
            img = rescale(image[i], (-1, 1), (0, 255), clamp=True)
        
        img = img.detach().to('cpu')
        img_pil = reverse_transforms(img)
        if timestep is not None:
            img_pil.save(f'{filename}_{timestep}.png')
        else:
            img_pil.save(f'{filename}_{i}.png')



def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
    
    
    
def call_step_and_decode_no_loop(timestep, latents, model_output, decoder, sampler, starting_time, FLAGS):
    latents = sampler.step(timestep.to(FLAGS.device), latents.to(FLAGS.device), model_output.to(FLAGS.device)).to(FLAGS.device)
    image = decoder(latents)
    
    if FLAGS.log:
        output_path = f"touch_results/diffusion/{starting_time}/image_test/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, f"timestep")
        
        save_image(image, output_path, timestep.item())
    
    
    

def call_step_and_decode_full_loop(timestep, latents, diffusion, context, decoder, sampler, epoch, starting_time, FLAGS):
    #! check this code
    # if FLAGS.generate_from_pure_noise:
    #     latents = torch.randn(latents.size(), device=FLAGS.device)
    # else:
    #     latents, _ = sampler.add_noise(latents, sampler.timesteps[0])
        
    latents = torch.randn(latents.size(), device=FLAGS.device)
    
    diffusion.eval()
    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        # torch.cuda.empty_cache()
        time_embedding = get_time_embedding(timestep).to(FLAGS.device)
        model_input = latents
        model_output = diffusion(model_input, context, time_embedding)
        latents = sampler.step(timestep.to(FLAGS.device), latents.to(FLAGS.device), model_output.to(FLAGS.device)).to(FLAGS.device)
    latents_for_decoding = latents.clone()
    diffusion.to("cpu")
    image = decoder(latents_for_decoding)
    
    if FLAGS.log:
        output_path = f"touch_results/diffusion/{starting_time}/full_loop"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, f"epoch_{epoch}")
            
        save_image(image, output_path)
    
    
    # TODO: implement the saving code for the context image
    # if FLAGS.conditioning != "uncondition":
    #     context = decoder(context.reshape(-1, 4, 32, 32))
        
    #     output_path = f"touch_results/diffusion/{starting_time}/context{epoch}"
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
            
    #     save_image(context, output_path)


def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return (20 * torch.log10(1.0 / torch.sqrt(mse)))

def gaussian_kernel(size: int, sigma: float):
    """Creates a 1D Gaussian kernel."""
    coords = torch.arange(size) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()

def create_window(window_size: int, sigma: float, channels: int):
    """Creates a 2D Gaussian filter window."""
    _1D_window = gaussian_kernel(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.T
    return _2D_window.expand(channels, 1, window_size, window_size)

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, C1: float = 0.01**2, C2: float = 0.03**2):
    """Computes the Structural Similarity Index (SSIM) between two images."""
    _, channels, _, _ = img1.shape
    window = create_window(window_size, sigma, channels).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(img1**2, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean()

