import torch
import torch.nn

from src.diffusion.ddpm import DDPMSampler
from src.utils.ranking_model import RankingEncoder
from src.utils.utils import *
from src.conditioning.rgbd_resnet import ResnetEncoder, ResnetRGBDEncoder


def validation(models, test_loader, starting_time, FLAGS):
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
    
    decoder.eval()
    context_encoder.eval()
    diffusion.eval()
    
    
    average_psnr_full = 0
    average_ssim_full = 0
    image_counter = 0
    with torch.no_grad():
        # with tqdm(test_loader, unit="batch") as t:
        for idx, (touch_image, rgbd_conditioning) in enumerate(test_loader):
            torch.cuda.empty_cache()
            decoder.to(FLAGS.idle_device)
            diffusion.to(FLAGS.device)
            context_encoder.to(FLAGS.device)
            
            # Define the noise to start denoising with the diffusion model
            latent_shape = (touch_image.shape[0], 4, FLAGS.image_size//8, FLAGS.image_size//8)
            starting_noise = torch.randn(latent_shape, generator=generator, device=FLAGS.device)
            
            # Define conditioning and ground truth data
            ground_truth = touch_image.to(FLAGS.device)
            rgbd_conditioning = rgbd_conditioning.to(FLAGS.device)
            context = context_encoder(rgbd_conditioning)
            context_encoder.to(FLAGS.idle_device)
            
            # timesteps = tqdm(sampler.timesteps)
            latents = starting_noise
            timesteps = tqdm(sampler.timesteps)
            for i, timestep in enumerate(timesteps):
                # torch.cuda.empty_cache()
                time_embedding = get_time_embedding(timestep).to(FLAGS.device)
                model_input = latents
                model_output = diffusion(model_input, context, time_embedding)
                latents = sampler.step(timestep.to(FLAGS.device), latents.to(FLAGS.device), model_output.to(FLAGS.device)).to(FLAGS.device)
            latents_for_decoding = latents.clone()
            diffusion.to(FLAGS.idle_device)
            decoder.to(FLAGS.device)
            image_pred = decoder(latents_for_decoding)
        
            for i in range(ground_truth.size(0)):
                # Compute metrics: PSNR
                psnr = compute_psnr(ground_truth[i], image_pred[i])
                print(f"PSNR {i}: {psnr}")
                
                # Compute metrics: SSIM
                ssim = compute_ssim(ground_truth[i].unsqueeze(0), image_pred[i].unsqueeze(0))
                print(f"SSIM {i}: {ssim}")
                
                average_psnr_full += psnr
                average_ssim_full += ssim
                image_counter += 1
            
            # TODO: Compute metrics: LPIPS
            
            # TODO: Compute metrics: FID
            
            # TODO: Compute metrics: Ranking Encoder
            
            # Save images: ground truth, context, predicted image
            if FLAGS.log:
                output_path = f"touch_evaluation/{starting_time}/{idx}/"
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    
                rgbd_conditioning_expanded = [rgbd_conditioning[:,0:3],
                                                rgbd_conditioning[:,3:4],
                                                rgbd_conditioning[:,4:7],
                                                rgbd_conditioning[:,7:8],
                                                rgbd_conditioning[:,8:12],
                                                rgbd_conditioning[:,12:15]]
                
                # results = torch.cat([ground_truth, torch.cat(rgbd_conditioning_expanded), image_pred], dim=1)
                
                save_image(ground_truth, os.path.join(output_path, "ground_truth"))
                for index, step in enumerate(FLAGS.cond_options):
                    save_image(rgbd_conditioning_expanded[index], os.path.join(output_path, f"ctx_{step}"), timestep=None, old_range=(0, 1), new_range=(0, 255))
                save_image(image_pred, os.path.join(output_path, "predicted_image"))
                
                # for index, step in enumerate(["ground_truth", f"{FLAGS.cond_options}", "predicted_image"]):
                #     save_image(results[:,index], os.path.join(output_path, step))
                
    print(f"Average PSNR: {average_psnr_full / image_counter}")
    print(f"Average SSIM: {average_ssim_full / image_counter}")
        