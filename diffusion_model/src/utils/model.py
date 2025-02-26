from src.autoencoder.encoder import VAE_Encoder
from src.autoencoder.decoder import VAE_Decoder
from src.diffusion.my_unet import Diffusion, SwitchSequential
from src.conditioning.rgbd_resnet import ResnetEncoder, ResnetRGBNEncoder, ResnetRGBDEncoder
from src.utils.model_converter import load_from_standard_weights

import torch
import torch.nn as nn


def preload_models(flags=None, conditioning_shape=0):
    assert flags is not None, "Flags must be provided to preload models"
    
    if flags.pretrained == "standard":
        # state_dict = load_from_standard_weights("weights/tarf_model/img2touch.ckpt", flags.device)
        state_dict = load_from_standard_weights("weights/tarf_model/img2touch.ckpt", flags.device, flags.pre_trained_model)
    
    
    # ---------------------
    # Load Encoder
    # ---------------------
    encoder = VAE_Encoder().to(flags.device)
    if flags.pretrained == "touch":
        #encoder.load_state_dict(torch.load("weights/autoencoder/encoder_2024-10-13-40-54.pth", weights_only=False))
        encoder.load_state_dict(torch.load("weights/touch/autoencoder/encoder_2025-02-19-11-30.pth", weights_only=False))
    
    
    # ---------------------
    # Load Decoder
    # ---------------------
    decoder = VAE_Decoder().to(flags.device)
    if flags.pretrained == "touch":
        #decoder.load_state_dict(torch.load("weights/autoencoder/decoder_2024-10-13-40-54.pth", weights_only=False))
        decoder.load_state_dict(torch.load("weights/touch/autoencoder/decoder_2025-02-19-11-30.pth", weights_only=False))


    # ---------------------
    # Load Diffusion
    # ---------------------
    diffusion = Diffusion(flags).to(flags.device)
    if flags.test:
        diffusion.load_state_dict(torch.load("weights/diffusion/diffusion_2024-12-04-16-51.pth", weights_only=False))



    # ---------------------
    # Load Conditining Encoder
    # ---------------------
    if flags.conditioning != "uncondition":
        if flags.cond_type == "rgbd":
            conditioner_model = ResnetRGBDEncoder(model_name=flags.model_name,conditioning_shape=conditioning_shape).to(flags.device)
        elif flags.cond_type == "nocs":
            conditioner_model = ResnetRGBNEncoder(model_name=flags.model_name,conditioning_shape=conditioning_shape).to(flags.device)
        else:
            conditioner_model = ResnetEncoder(model_name=flags.model_name).to(flags.device)
    else:
        conditioner_model = None

    return {
        'encoder': encoder,
        'decoder': decoder,
        'conditioner': conditioner_model,
        'diffusion': diffusion,
    }