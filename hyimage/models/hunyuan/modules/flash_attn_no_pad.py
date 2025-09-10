import torch
import torch.nn.functional as F
from einops import rearrange

def get_cu_seqlens(text_mask: torch.Tensor, img_len: int):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    # Vectorized computation for s and the indices
    s = text_len + img_len
    
    # Create the base indices for the tensor operations
    batch_indices = torch.arange(batch_size, device=text_mask.device)
    
    # Calculate s1 and s2 for all batches at once
    s1 = batch_indices * max_len + s
    s2 = (batch_indices + 1) * max_len
    
    # Interleave s1 and s2
    interleaved = torch.stack([s1, s2], dim=1).flatten()
    
    # Prepend the initial 0
    cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32, device=text_mask.device), interleaved])

    return cu_seqlens, max_len

def create_attention_mask(cu_seqlens: torch.Tensor, max_s: int, causal: bool = False):
    batch_size = (len(cu_seqlens) - 1) // 2
    device = cu_seqlens.device
    
    mask = torch.zeros(batch_size, max_s, max_s, dtype=torch.bool, device=device)
    
    if causal:
        base_causal_mask = torch.tril(torch.ones(max_s, max_s, dtype=torch.bool, device=device))
    
    for i in range(batch_size):
        start_idx = cu_seqlens[2 * i].item()
        end_idx = cu_seqlens[2 * i + 1].item()
        seq_len = end_idx - start_idx
        
        if causal:
            mask[i, :seq_len, :seq_len] = base_causal_mask[:seq_len, :seq_len].clone()
        else:
            mask[i, :seq_len, :seq_len] = True
    
    return mask

def torch_sdpa_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_s: int,
    causal: bool = False,
    deterministic: bool = False,
):
    batch_size, seq_len, n_heads, head_dim = q.shape
    
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    attn_mask = create_attention_mask(cu_seqlens, max_s, causal)
    attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
    
    with torch.no_grad():
        attn_mask = attn_mask.clone()
    
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=False,
    )
    
    output = output.transpose(1, 2).contiguous()
    
    return output

def torch_sdpa_no_pad(
    qkv: torch.Tensor,
    key_padding_mask: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale=None,
    deterministic: bool = False,
):
    batch_size, seq_len, _, n_heads, head_dim = qkv.shape
    
    q, k, v = qkv.unbind(dim=2)
    
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    key_padding_mask = key_padding_mask.bool()
    
    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).clone()
    attn_mask = attn_mask.repeat(1, 1, seq_len, 1)
    
    query_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).clone()
    query_mask = query_mask.repeat(1, 1, 1, seq_len)
    
    attn_mask = torch.logical_and(attn_mask, query_mask)
    
    if causal:
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=qkv.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        attn_mask = torch.logical_and(attn_mask, causal_mask)
    
    attn_mask = attn_mask.repeat(1, n_heads, 1, 1)
    
    with torch.no_grad():
        attn_mask = attn_mask.clone()
    
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        scale=softmax_scale,
        is_causal=False,
    )
    
    output = output.transpose(1, 2).contiguous()
    
    return output

def cleanup_attention_cache():
    if hasattr(torch.backends.cuda, 'sdp_kernel'):
        torch.backends.cuda.sdp_kernel(enable_flash=False)
        torch.backends.cuda.sdp_kernel(enable_flash=True)
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

flash_attn_v3 = torch_sdpa_v3
flash_attn_no_pad = torch_sdpa_no_pad
