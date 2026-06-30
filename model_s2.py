import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

# 引入 Diffusers 库中的 VAE (需要安装: pip install diffusers)
from diffusers import AutoencoderKL

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==========================================
# 1. DiT Block (核心模块)
# ==========================================
class DiTBlock(nn.Module):
    """
    DiT Block with adaptive Layer Norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# ==========================================
# 2. Positional Embedding
# ==========================================
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# ==========================================
# 3. Sky Encoder (风格/光照编码器)
# ==========================================
class SkyEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=1024):
        super().__init__()
        # 一个简单的类 ResNet 结构，将天空图编码为向量
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), # 128x256
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1), # 64x128
            
            self._make_layer(64, 128, 2), # 32x64
            self._make_layer(128, 256, 2), # 16x32
            self._make_layer(256, 512, 2), # 8x16
            self._make_layer(512, hidden_size, 2) # 4x8
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _make_layer(self, in_c, out_c, blocks):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        x = self.mlp(x)
        return x

# ==========================================
# 4. Geometry Encoder (几何信息编码器)
# ==========================================
class GeoLatentEncoder(nn.Module):
    """
    将 2通道 (Opacity, Depth) 压缩到 VAE Latent 相同的空间维度 (H/8, W/8)
    """
    def __init__(self, in_channels=2, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(), # /2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(), # /4
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1), # /8
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 5. SatDiT 主模型
# ==========================================
class SatDiT(nn.Module):
    def __init__(
        self,
        input_size=(32, 64), # Latent size (256/8, 512/8)
        patch_size=2,
        in_channels=4, # VAE latent channels
        cond_channels=5, # 3 RGB + 1 Opacity + 1 Depth
        hidden_size=1024, # DiT-L
        depth=24,         # DiT-L/2 original is 28, but 24 is also common. Let's use 28 for pretrained.
        num_heads=16,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        # 1. 编码器部分
        # Sky Encoder
        self.sky_encoder = SkyEncoder(in_channels=3, hidden_size=hidden_size)
        
        # Geo Encoder (处理 Opacity + Depth) -> Latent Space
        self.geo_encoder = GeoLatentEncoder(in_channels=2, out_channels=4)

        # 2. DiT 主干
        # 计算输入通道总数: Noisy Latent (4) + Input RGB Latent (4) + Input Geo Latent (4)
        # Input RGB Latent 来自 VAE，Geo 来自 GeoEncoder
        self.total_input_channels = in_channels + 4 + 4 
        
        self.x_embedder = PatchEmbed(input_size, patch_size, self.total_input_channels, hidden_size, bias=True)
        self.t_embedder = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # 位置编码 (Learned or Fixed, here using fixed sincos for easy extension)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=4.0) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # 初始化 Patch Embedder 和 Blocks
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias.data, 0)
        
        # 初始化 pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(32/self.patch_size), int(64/self.patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化输出层 (Zero init)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = 32 // p
        w = 64 // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, sky_image, input_5ch_latents):
        """
        x: Noisy Latents (B, 4, 32, 64)
        t: Timesteps (B,)
        sky_image: (B, 3, 256, 512)
        input_5ch_latents: 包含 RGB Latents (B, 4, 32, 64) 和 Geo Latents (B, 4, 32, 64)
        """
        # 1. Prepare Conditions
        # (B, 3, 256, 512) -> (B, hidden_size)
        c = self.sky_encoder(sky_image) 
        
        # Time Embedding
        t_freq = timestep_embedding(t, 256)
        t_emb = self.t_embedder(t_freq)
        
        # 这里的 condition 是 Sky + Time
        c = t_emb + c 

        # 2. Concatenate Inputs in Latent Space
        # x: Noisy latents (4ch)
        # input_5ch_latents: (8ch) -> RGB VAE Latents (4) + Geo Encoded Latents (4)
        x_in = torch.cat([x, input_5ch_latents], dim=1) # (B, 12, 32, 64)

        # 3. DiT Forward
        x = self.x_embedder(x_in) + self.pos_embed
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        
        x = self.unpatchify(x)
        return x

# def timestep_embedding(timesteps, dim, max_period=10000):
#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#     ).to(device=timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # --- 【关键修复开始】 ---
    # 如果传入的是标量(0维)，先将其升维成1维张量 (1,)
    if len(timesteps.shape) == 0:
        timesteps = timesteps.unsqueeze(0)
    # --- 【关键修复结束】 ---

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
# ==========================================
# 6. 预训练权重加载工具
# ==========================================
def load_pretrained_dit_weights(model, model_name='DiT-L-2-256'):
    """
    从 Facebook 官方或 HuggingFace 加载 DiT 权重并适配到 SatDiT
    """
    print(f"Loading pretrained weights for {model_name}...")
    try:
        # 这里尝试直接加载 facebook/DiT-XL-2-256 的权重作为示例
        # 实际情况你可以下载 .pt 文件
        # URL: https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256/DiT-XL-2-256.pt
        state_dict = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/DiT/models/DiT-L-2-256/DiT-L-2-256.pt", 
            map_location='cpu'
        )
    except:
        print("Failed to download/load weights. Using random init.")
        return

    model_dict = model.state_dict()
    
    # 过滤和适配权重
    for k, v in state_dict.items():
        if k not in model_dict:
            continue
            
        # 1. 适配 Positional Embeddings (插值)
        if 'pos_embed' in k:
            # v shape: [1, 256, 1024] -> Grid 16x16
            # target shape: [1, 512, 1024] -> Grid 32x64
            v_grid = v.reshape(1, 16, 16, -1).permute(0, 3, 1, 2)
            v_new = F.interpolate(v_grid, size=(32, 64), mode='bicubic', align_corners=False)
            v_new = v_new.permute(0, 2, 3, 1).reshape(1, 32*64, -1)
            model_dict[k].copy_(v_new)
            print(f"Interpolated pos_embed from {v.shape} to {v_new.shape}")
            
        # 2. 适配 Input Projection (PatchEmbed)
        elif 'x_embedder.proj.weight' in k:
            # v shape: [1024, 4, 2, 2]
            # target: [1024, 12, 2, 2]
            new_weight = model_dict[k].clone()
            # 复制前4通道 (Noisy Latents)
            new_weight[:, :4, :, :] = v
            # 其他通道 (Conditions) 保持初始化状态 (通常接近0或Xavier)
            # 建议将新增通道初始化为0，以便一开始模型表现得像无条件模型
            new_weight[:, 4:, :, :] = 0 
            model_dict[k].copy_(new_weight)
            print(f"Adapted input conv from {v.shape} to {new_weight.shape}")

        # 3. 跳过 Class Embedding (我们用 SkyEncoder)
        elif 'y_embedder' in k:
            continue
            
        # 4. 其他直接加载 (Blocks, Final Layer等)
        else:
            if v.shape == model_dict[k].shape:
                model_dict[k].copy_(v)
            else:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")
    
    print("Pretrained weights loaded successfully.")