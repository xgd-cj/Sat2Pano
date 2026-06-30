import os
import random  # <--- 修改点 1: 引入 random 库
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image
from tqdm import tqdm
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# 引入你的模型定义
# 请确保 model_s2.py 在当前目录下，或者在 PYTHONPATH 中
from model_s2 import SatDiT, load_pretrained_dit_weights

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==========================================
# 1. Dataset (修改后：天空图随机选取)
# ==========================================
class SatPanoramaDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.dirs = {
            'pano': os.path.join(data_root, 'init_proj'),
            'opacity': os.path.join(data_root, 'opacity'),
            'depth': os.path.join(data_root, 'depth'),
            'gt': os.path.join(data_root, 'ground_truth'),
            'sky': os.path.join(data_root, 'sky')
        }
        
        # 1. 主索引：基于 GT 文件夹，确保文件名一致性
        self.filenames = [f for f in os.listdir(self.dirs['gt']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.filenames.sort()

        # 2. 天空池：获取所有可用的天空图文件名
        # <--- 修改点 2: 预先加载所有天空文件名
        if os.path.exists(self.dirs['sky']):
            self.sky_filenames = [f for f in os.listdir(self.dirs['sky']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.sky_filenames = []

        self.init_pano_transform = transforms.Compose([
            transforms.Resize((256, 512)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        self.geo_transform = transforms.Compose([
            transforms.Resize((256, 512)), transforms.Grayscale(1), transforms.ToTensor()])
        self.pano_transform = transforms.Compose([
            transforms.Resize((256, 512)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        # 这里的 filename 保证了 gt, init_pano, opacity, depth 是一一对应的
        filename = self.filenames[idx]
        
        def load_img(path, gray=False):
            if not os.path.exists(path): return None
            return Image.open(path).convert('L' if gray else 'RGB')

        # 加载 GT
        gt_img = self.pano_transform(load_img(os.path.join(self.dirs['gt'], filename)))
        
        # <--- 修改点 3: 随机选取天空图
        # 不再使用 filename，而是随机从池子里选
        if len(self.sky_filenames) > 0:
            random_sky_name = random.choice(self.sky_filenames)
            sky_path = os.path.join(self.dirs['sky'], random_sky_name)
        else:
            sky_path = "" # 路径不存在，后续逻辑会处理生成全黑图

        # 如果随机到的天空图存在则加载，否则生成全黑图
        if os.path.exists(sky_path):
            sky_pil = load_img(sky_path)
            sky_img = self.pano_transform(sky_pil)
        else:
            sky_img = self.pano_transform(Image.new('RGB', (512, 256)))

        # 加载初始投影图 (同名)
        init_pano_path = os.path.join(self.dirs['pano'], filename)
        init_pano = self.init_pano_transform(load_img(init_pano_path))
        
        # 加载 Opacity (同名)
        op = load_img(os.path.join(self.dirs['opacity'], filename), True)
        opacity = self.geo_transform(op) if op else torch.zeros(1, 256, 512)
        
        # 加载 Depth (同名)
        dep = load_img(os.path.join(self.dirs['depth'], filename), True)
        depth = self.geo_transform(dep) if dep else torch.zeros(1, 256, 512)
        
        # 拼接 5 通道输入
        input_5ch = torch.cat([init_pano, opacity, depth], dim=0)
        
        return {'gt_image': gt_img, 'input_5ch': input_5ch, 'sky_image': sky_img}

# ==========================================
# 2. 配置
# ==========================================
TRAIN_ROOT = "data/CVACT_stage2train"
VAL_ROOT   = "data/CVACT_stage2val"
SAVE_DIR   = "outputs/stage2_rand"

LR = 1e-4                  
BATCH_SIZE = 8 # DiT 显存占用较大，可能需要调小 BS
EPOCHS = 300
NUM_WORKERS = 4
HIDDEN_SIZE = 1024 # DiT-L

# DDP
MASTER_ADDR = 'localhost'
MASTER_PORT = '12366'

# ==========================================
# 3. 工具函数
# ==========================================
@torch.no_grad()
def encode_latents(vae, image):
    # image: [B, 3, 256, 512], range [-1, 1]
    latents = vae.encode(image).latent_dist.sample()
    latents = latents * 0.18215 # Scaling factor for SD VAE
    return latents

@torch.no_grad()
def decode_latents(vae, latents):
    latents = latents / 0.18215
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

# ==========================================
# 4. 训练 Step
# ==========================================
def train_one_epoch(model, vae, loader, optimizer, scheduler, device, rank):
    model.train()
    vae.eval() # VAE 始终冻结
    
    if rank == 0:
        pbar = tqdm(loader, desc="Training DiT", dynamic_ncols=True)
    else:
        pbar = loader
        
    loss_accum = 0.0
    
    for batch in pbar:
        # 数据准备
        gt_img = batch['gt_image'].to(device) # [-1, 1]
        input_5ch = batch['input_5ch'].to(device) # [B, 5, H, W]
        sky_img = batch['sky_image'].to(device) # [B, 3, H, W]
        
        bs = gt_img.shape[0]

        # 1. VAE Encode GT -> Latents (Target)
        latents = encode_latents(vae, gt_img)
        
        # 2. Prepare Condition Latents
        input_rgb = input_5ch[:, :3, :, :] # RGB 部分
        input_geo = input_5ch[:, 3:, :, :] # Opacity+Depth 部分
        
        # RGB 部分走 VAE 编码
        cond_rgb_latents = encode_latents(vae, input_rgb) # [B, 4, 32, 64]
        
        # Geo 部分走模型内部的 GeoEncoder
        geo_latents = model.module.geo_encoder(input_geo) # [B, 4, 32, 64]
        
        # 拼接 Condition Latents
        input_5ch_latents = torch.cat([cond_rgb_latents, geo_latents], dim=1) # [B, 8, 32, 64]

        # 3. Add Noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # 4. Forward
        # SatDiT forward(x, t, sky, cond_latents)
        model_pred = model(noisy_latents, timesteps, sky_img, input_5ch_latents)
        
        # 如果模型输出包含 variance (8通道)，只取前4通道计算 loss
        if model.module.out_channels == 8:
            model_pred, _ = model_pred.chunk(2, dim=1)
            
        # 5. Loss
        loss = F.mse_loss(model_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_accum += loss.item()
        
        if rank == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # DDP Reduce Loss
    loss_tensor = torch.tensor(loss_accum, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item() / (len(loader) * dist.get_world_size())

# ==========================================
# 5. 主程序
# ==========================================
def main_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"[Init] DiT Training with SD-VAE. Rank: {rank}")

    # 1. Load VAE (Frozen)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    
    # 2. Init DiT Model
    model = SatDiT(
        input_size=(32, 64), 
        patch_size=2, 
        in_channels=4, 
        cond_channels=5,
        hidden_size=1024, 
        depth=28, 
        num_heads=16
    ).to(device)
    
    # 加载预训练权重
    load_pretrained_dit_weights(model, model_name='DiT-L-2-256')
    
    model = DDP(model, device_ids=[rank])
    
    # 3. Setup Scheduler
    scheduler = DDPMScheduler.from_pretrained("facebook/DiT-XL-2-256", subfolder="scheduler") 

    # 4. Data
    train_dataset = SatPanoramaDataset(TRAIN_ROOT)
    # 确保 drop_last=True 防止 batch size 不一致导致 DDP 卡住
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, drop_last=True)
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    
    # 6. Loop
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, vae, train_loader, optimizer, scheduler, device, rank)
        
        if rank == 0:
            print(f"Epoch {epoch+1} Done. Loss: {avg_loss:.4f}")
            
            # Save Checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, "latest_dit.pth"))
                
                # --- Sampling / Visualization ---
                model.eval()
                with torch.no_grad():
                    # 采样一张图片
                    sample_batch = next(iter(train_loader)) 
                    input_5ch = sample_batch['input_5ch'][:1].to(device)
                    sky_img = sample_batch['sky_image'][:1].to(device)
                    
                    # Encode Condition
                    input_rgb = input_5ch[:, :3]
                    input_geo = input_5ch[:, 3:]
                    cond_rgb = encode_latents(vae, input_rgb)
                    geo_latents = model.module.geo_encoder(input_geo)
                    cond_latents = torch.cat([cond_rgb, geo_latents], dim=1)
                    
                    # Diffusion Sampling Loop
                    latents = torch.randn(1, 4, 32, 64).to(device)
                    
                    scheduler.set_timesteps(50) 
                    for t in tqdm(scheduler.timesteps, desc="Sampling"):
                        model_output = model(latents, t, sky_img, cond_latents)
                        if model.module.out_channels == 8:
                            model_output, _ = model_output.chunk(2, dim=1)
                            
                        latents = scheduler.step(model_output, t, latents).prev_sample
                        
                    img_pred = decode_latents(vae, latents)
                    save_image(img_pred, os.path.join(SAVE_DIR, f"sample_epoch_{epoch+1}.png"))

    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 SatDiT with random sky conditioning.")
    parser.add_argument("--train_root", type=str, default=TRAIN_ROOT, help="Stage 2 training data root.")
    parser.add_argument("--val_root", type=str, default=VAL_ROOT, help="Reserved validation data root.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Directory for checkpoints and samples.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n_gpus", type=int, default=None, help="Number of GPUs to use. Defaults to all visible GPUs.")
    parser.add_argument("--master_port", type=str, default=MASTER_PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TRAIN_ROOT = args.train_root
    VAL_ROOT = args.val_root
    SAVE_DIR = args.save_dir
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LR = args.lr
    MASTER_PORT = args.master_port

    n_gpus = torch.cuda.device_count()
    if args.n_gpus is not None:
        n_gpus = min(args.n_gpus, n_gpus)
    if n_gpus < 1:
        raise RuntimeError("Stage 2 training requires at least one CUDA GPU.")
    mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus,))
