import torch
import torch.nn.functional as F
from einops import rearrange


def get_cu_seqlens(text_mask: torch.Tensor, img_len: int):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len
    
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=text_mask.device)
    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2
    
    return cu_seqlens, max_len


def create_attention_mask(cu_seqlens: torch.Tensor, max_s: int, causal: bool = False):
    batch_size = (len(cu_seqlens) - 1) // 2
    device = cu_seqlens.device
    
    mask = torch.zeros(batch_size, max_s, max_s, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        start_idx = cu_seqlens[2 * i]
        end_idx = cu_seqlens[2 * i + 1]
        seq_len = end_idx - start_idx
        
        mask[i, :seq_len, :seq_len] = True
        
        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            mask[i, :seq_len, :seq_len] = mask[i, :seq_len, :seq_len] & causal_mask
    
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
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    attn_mask = create_attention_mask(cu_seqlens, max_s, causal)
    attn_mask = attn_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
    
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=False,
    )
    
    output = output.transpose(1, 2)
    
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
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
    attn_mask = attn_mask.expand(batch_size, 1, seq_len, seq_len)
    
    query_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)
    attn_mask = attn_mask & query_mask.expand(batch_size, 1, seq_len, seq_len)
    
    if causal:
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=qkv.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attn_mask = attn_mask & causal_mask
    
    attn_mask = attn_mask.expand(batch_size, n_heads, seq_len, seq_len)
    
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if q.training else 0.0,
        scale=softmax_scale,
        is_causal=False,
    )
    
    output = output.transpose(1, 2)
    
    return output


flash_attn_v3 = torch_sdpa_v3
flash_attn_no_pad = torch_sdpa_no_pad
