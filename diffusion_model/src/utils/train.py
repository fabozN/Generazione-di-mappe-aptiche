import os

import torch
from tqdm import tqdm
import wandb

from src.utils.utils import *
from src.diffusion.ddpm import DDPMSampler
from src.conditioning.rgbd_resnet import ResnetEncoder, ResnetRGBDEncoder



def train(models, optimizer, scheduler, train_loader, criterion, starting_time, FLAGS):
    encoder = models["encoder"].to(FLAGS.device)
    decoder = models["decoder"].to(FLAGS.device)
    context_encoder = models["conditioner"].to(FLAGS.device)
    diffusion = models["diffusion"].to(FLAGS.device)
    
    generator = torch.Generator(device=FLAGS.device)
    if FLAGS.seed is not None:
        generator.manual_seed(FLAGS.seed)
        
    if FLAGS.sampler_name == "ddpm":
        sampler = DDPMSampler(generator, num_training_steps=FLAGS.num_training_steps)
        sampler.set_inference_timesteps(FLAGS.num_inference_steps)
    else:
        raise ValueError(f"Sampler {FLAGS.sampler_name} not supported")
    
    encoder.eval()
    decoder.eval()
    context_encoder.eval()
    
    # start tqdm
    for epoch in range(FLAGS.epochs):
        diffusion.train()
        diffusion.to(FLAGS.device)
        decoder.to(FLAGS.idle_device)
        
        total_epoch_loss = 0
        timestep_mean = 0
        with tqdm(train_loader, desc=f'Epoch {epoch}', unit="batch") as t:
            for i, (touch_image, rgbd_conditioning) in enumerate(t):
                torch.cuda.empty_cache()
                encoder.to(FLAGS.device)
                context_encoder.to(FLAGS.device)
                
                
                latent_shape = (touch_image.shape[0], 4, FLAGS.image_size//8, FLAGS.image_size//8)
                optimizer.zero_grad()
                
                # torch.cuda.empty_cache()
                image = touch_image.to(FLAGS.device)
                context = rgbd_conditioning.to(FLAGS.device)
                
                if FLAGS.conditioning == "uncondition":
                    print("Unconditioned model not supported")
                else:
                    # Encode input data to get latents
                    encoder_noise = torch.randn(latent_shape, generator=generator, device=FLAGS.device)
                    latents_no_noise = encoder(image, encoder_noise)
                    
                    # Get time embedding to pass to the diffusion model
                    timestep_idx = torch.randint(0, len(sampler.training_timesteps), (1,)) 
                    timestep = sampler.training_timesteps[timestep_idx]
                    time_embedding = get_time_embedding(timestep).to(FLAGS.device)
                    
                    # Add noise to the latent representation
                    noisy_latent, inj_noise = sampler.add_noise(latents_no_noise, timestep)
                    
                    # Get the context representation
                    context_latent = context_encoder(context)
                    
                    # Pass into diffusion
                    model_output = diffusion(noisy_latent, context_latent, time_embedding)
                    
                    
                    # Calculate loss and backpropagate
                    loss = criterion(model_output, inj_noise)
                    loss.backward()
                    optimizer.step()
                    
                    encoder.to(FLAGS.idle_device)
                    context_encoder.to(FLAGS.idle_device)
                    
                    timestep_mean += timestep.item()
                    total_epoch_loss += loss.item()
                    t.set_postfix(loss=total_epoch_loss/(i+1), t=timestep_mean//(i+1))
                    
                    if FLAGS.scheduling_lr:
                        scheduler.step()    
                    
        # wandb log
        if FLAGS.wandb:
            total_epoch_loss /= len(train_loader)
            wandb.log({"loss": total_epoch_loss, "epoch": epoch})
        
        # Log the output
        decoder.to(FLAGS.device)
        # call_step_and_decode_no_loop(timestep, noisy_latent, model_output, decoder, sampler, starting_time, FLAGS)
        if epoch % FLAGS.log_interval == 0 or epoch == FLAGS.epochs-1:
            call_step_and_decode_full_loop(timestep, latents_no_noise[0].unsqueeze(0), diffusion, context_latent[0].unsqueeze(0), decoder, sampler, epoch, starting_time, FLAGS)
            
        # Save the model
        if FLAGS.log:
            if epoch % FLAGS.save_interval == 0 or epoch == FLAGS.epochs-1:
                weights_path = 'weights/diffusion'
                if not os.path.exists(weights_path):
                    os.makedirs(weights_path)
                torch.save(diffusion.state_dict(), f'{weights_path}/diffusion_{starting_time}.pth')
                print("Saved model :)")