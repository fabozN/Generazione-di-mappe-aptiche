import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_Size, Sequence_Length, Dimension)
        
        input_shape = x.shape
        batch_size, sequence_length, dimension = input_shape
        
        intermin_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (Batch_Size, Sequence_Length, Dimension) --> (Batch_Size, Sequence_Length, 3 * Dimension)
        # --> 3 tensors of shape (Batch_Size, Sequence_Length, Dimension)
        
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Sequence_Length, Dimension) --> (Batch_Size, Sequence_Length, n_Head, d_Head / n_Head)
        # --> (Batch_Size, n_Head, Sequence_Length, d_Head / n_Head)
        q = q.view(*intermin_shape).transpose(1, 2)
        k = k.view(*intermin_shape).transpose(1, 2)
        v = v.view(*intermin_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-2, -1)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, n_Head, Sequence_Length, d_Head / n_Head) @ (Batch_Size, n_Head, d_Head / n_Head, Sequence_Length)
        # --> (Batch_Size, n_Head, Sequence_Length, Dimension / n_Head)
        output = weight @ v
        
        # (Batch_Size, n_Head, Sequence_Length, Dimension / n_Head) --> (Batch_Size, Sequence_Length, n_Head, Dimension / n_Head)
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (Batch_Size, Sequence_Length, Dimension)
        return output
    

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_embed = d_embed
        
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, causal_mask=False):
        # x: (latent): (Batch_Size, Sequence_Length_Q, Dimension_Q)
        # y: (context): (Batch_Size, Sequence_Length_KV, Dimension_KV)
        
        input_shape_x = x.shape
        input_shape_y = y.shape
        
        batch_size, sequence_length_x, dimension_x = input_shape_x
        _, sequence_length_y, _, _ = input_shape_y
        # _, sequence_length_y = input_shape_y
        
        intermin_shape_x = (batch_size, sequence_length_x, self.n_heads, self.d_head)
        intermin_shape_y = (batch_size, sequence_length_y, self.n_heads, self.d_head)
        
        # test = torch.rand((1,5,320)).to(y.device)
        
        q = self.q_proj(x).view(intermin_shape_x).transpose(1, 2)
        k = self.k_proj(y).view(intermin_shape_y).transpose(1, 2)
        v = self.v_proj(y).view(intermin_shape_y).transpose(1, 2)
        # k = self.k_proj(y).view(intermin_shape_x).transpose(1, 2)
        # v = self.v_proj(y).view(intermin_shape_x).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        
        output = output.transpose(1, 2).contiguous()
        
        output = output.view(input_shape_x)
        
        output = self.out_proj(output)
        
        return output