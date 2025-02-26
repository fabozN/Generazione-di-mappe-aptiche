import torch
import torch.nn as nn
import torch.nn.functional as F
from src.autoencoder.encoder import VAE_Encoder
from src.autoencoder.decoder import VAE_AttentionBlock, VAE_ResidualBlock

from src.utils.attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        
        x = self.linear_1(x)
        
        x = F.silu(x)
        
        x = self.linear_2(x)
        
        # (1, 1280)
        return x
    

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (Batch_Size, in_channels, Height, Width)
        # time: (1, 1200)
        
        residue = feature
        
        feature = self.groupnorm_feature(feature)
        
        feature = F.silu(feature)
        
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        
        merged = F.silu(merged)
        
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embd: int, d_context=1024, cond=False):
        super().__init__()
        self.cond = cond
        channels = n_heads * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        if cond:
            self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False) 
        else:
            self.attention_2 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, features, Height, Width)
        # context: (Batch_Size, Sequence_Length, Dimension)
        
        residue_long = x
        
        x = self.groupnorm(x)
        
        #! video code showed this but git code not
        # x = F.silu(x)
        
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, features, Height, Width) --> (Batch_Size, features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, features, Height * Width) --> (Batch_Size, Height * Width, features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection
        residue_short = x
        
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short
        
        residue_short = x
        
        # Normalization + Cross-Attention with skip connection
        x = self.layernorm_2(x)
        
        # Cross-Attention
        if self.cond:            
            x = self.attention_2(x, context)
        else:
            x = self.attention_2(x)
        
        x = x + residue_short
        
        residue_short = x
        
        # Normalization + FF with GeGLU with skip connection
        
        x = self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        
        #! video code showed F.gelu(x) but git code not
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        # x = F.gelu(x)
        
        x = self.linear_geglu_2(x)
        
        x = x + residue_short
        
        # (Batch_Size, Height * Width, features) --> (Batch_Size, features, Height * Width)
        x = x.transpose(-1, -2)
        
        x = x.view((n, c, h, w))
        
        x = self.conv_output(x)
        
        return x + residue_long

    

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) --> (Batch_Size, Features, Height*2, Width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x    

    

class UNET_OutputLayer(nn.Module):
    # 2:38:44 https://www.youtube.com/watch?v=ZBKpAp_6TGI
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 320, Height/8, Width/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # (Batch_Size, 4, Height/8, Width/8)
        return x

    
class UNET(nn.Module):
    def __init__(self, flags=None):
        super().__init__()
        self.cross_attn = False
        self.input_channels = 4
        
        assert flags.conditioning in ['concat', 'attention', 'uncondition'], "Conditioning type must be concat, attention or uncondition"
        if flags.conditioning == 'concat':
            self.input_channels = 68
        elif flags.conditioning == 'attention':
            self.cross_attn = True
        elif flags.conditioning == 'uncondition':
            print("Unconditioned model")
        
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height/8, Width/8)
            SwitchSequential(nn.Conv2d(self.input_channels, 320, kernel_size=3, stride=1, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40, cond=self.cross_attn)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40, cond=self.cross_attn)),
            
            # (Batch_Size, 320, Height/8, Width/8) --> (Batch_Size, 640, Height/16, Width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80, cond=self.cross_attn)),
            
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80, cond=self.cross_attn)),
            
            # (Batch_Size, 640, Height/16, Width/16) --> (Batch_Size, 1280, Height/32, Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160, cond=self.cross_attn)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160, cond=self.cross_attn)),
            
            # (Batch_Size, 1280, Height/32, Width/32) --> (Batch_Size, 1280, Height/64, Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height/64, Width/64) --> (Batch_Size, 1280, Height/64, Width/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )
        
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, cond=self.cross_attn)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, cond=self.cross_attn)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160, cond=self.cross_attn), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80, cond=self.cross_attn), UpSample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40, cond=self.cross_attn)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, cond=self.cross_attn)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, cond=self.cross_attn))
        ])
        
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x



class Diffusion(nn.Module):
    def __init__(self, flags=None):
        super().__init__()
        assert flags is not None, "Flags must be provided to preload models"
        self.conditioning = flags.conditioning
        
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET(flags)
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_Size, 4, Height/8, Width/8)
        # context: (Batch_Size, Sequence_Length, Dimension)
        # time: (1, 320)
        
        # (1, 320) --> (1, 1280)
        time  = self.time_embedding(time)
        
        # (Batch_Size, 4, Height/8, Width/8) --> (Batch_Size, 320, Height/8, Width/8)
        if self.conditioning == 'concat':
            # (Batch_Size, 4, Height/8, Width/8) --> (Batch_Size, 68, Height/8, Width/8)
            latent = torch.cat((latent, context), dim=1)
        output = self.unet(latent, context, time)
        
        # (Batch_Size, 320, Height/8, Width/8) --> (Batch_Size, 4, Height/8, Width/8)
        output = self.final(output)
        
        # (Batch_Size, 4, Height/8, Width/8)
        return output
        