import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image
from tqdm import tqdm
import re

# 引入你的模型定义
from model_s2 import SatDiT

# 设置 HF 镜像（如果需要下载配置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    # --- 新增参数：序列模式开关 ---
    # True:  序列模式 (文件名格式: SatID_SeqID_rgb.png) -> 此时天空图取 SatID.png
    # False: 单图模式 (文件名格式: SatID_rgb.png)       -> 此时天空图取 SatID.png
    SEQ_MODE = True  
    
    # 你的输入文件夹（包含 _rgb.png, _depth.png, _opacity.png）
    INPUT_DIR = "outputs/video/S1" 
    
    # 你的天空图文件夹（包含 卫星ID.png）
    SKY_DIR = "data/CVUSA_train/train/sky_masks"
    
    # 结果保存文件夹
    OUTPUT_DIR = "outputs/video/S2"
    
    # 训练好的权重路径
    CHECKPOINT_PATH = "checkpoints/stage2/latest_dit.pth" 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = (256, 512) # (H, W)
    BATCH_SIZE = 1  
    NUM_INFERENCE_STEPS = 1000 

cfg = Config()

# ==========================================
# 2. 推理专用 Dataset (包含逻辑修改)
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, input_dir, sky_dir, seq_mode=True):
        super().__init__()
        self.input_dir = input_dir
        self.sky_dir = sky_dir
        self.seq_mode = seq_mode  # 保存模式状态
        
        # 1. 扫描所有初始渲染RGB图作为基准
        self.files = [f for f in os.listdir(input_dir) if f.endswith('_rgb.png')]
        self.files.sort()
        
        # 预处理转换
        self.norm_transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        self.geo_transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_SIZE),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def parse_filename(self, filename):
        """
        根据 SEQ_MODE 解析文件名
        """
        # 去掉后缀 _rgb.png 得到纯文件名部分
        base_name = filename.replace('_rgb.png', '')
        
        if self.seq_mode:
            # === 序列模式 (True) ===
            # 格式: SatID_SeqID_rgb.png
            # 逻辑: 从右往左找第一个 '_', 分离 SeqID 和 SatID
            if '_' in base_name:
                sat_id, seq_id = base_name.rsplit('_', 1)
            else:
                # 容错
                sat_id = base_name
                seq_id = "000"
            
            # base_prefix 用于寻找同组的 depth/opacity (包含 seq_id)
            base_prefix = base_name 
            
        else:
            # === 单图模式 (False) ===
            # 格式: SatID_rgb.png (即 base_name 就是 SatID)
            # 逻辑: 整个 base_name 就是 SatID，不需要剥离序列号
            sat_id = base_name
            seq_id = "single"  # 占位符，不影响逻辑
            
            # base_prefix 用于寻找同组的 depth/opacity (就是 SatID)
            base_prefix = base_name

        return sat_id, seq_id, base_prefix

    def __getitem__(self, idx):
        rgb_filename = self.files[idx]
        
        # 调用修改后的解析逻辑
        sat_id, seq_id, base_prefix = self.parse_filename(rgb_filename)
        
        # 构造文件路径
        # 注意: 无论是序列模式还是单图模式，depth/opacity 的前缀都应该和 rgb 的前缀一致
        # 序列模式: SatID_SeqID_depth.png
        # 单图模式: SatID_depth.png
        rgb_path = os.path.join(self.input_dir, rgb_filename)
        depth_path = os.path.join(self.input_dir, f"{base_prefix}_depth.png")
        opacity_path = os.path.join(self.input_dir, f"{base_prefix}_opacity.png")
        
        # 天空图路径: 始终是 SatID.png (不带序列号)
        sky_path = os.path.join(self.sky_dir, f"{sat_id}.png")
        #sky_path = os.path.join(self.sky_dir, f"{sat_id}.jpg")
        
        # --- 加载图片 (逻辑不变) ---
        # 1. Init RGB
        init_rgb = Image.open(rgb_path).convert('RGB')
        init_rgb = self.norm_transform(init_rgb)
        
        # 2. Opacity
        if os.path.exists(opacity_path):
            op = Image.open(opacity_path)
            opacity = self.geo_transform(op)
        else:
            print(f"Warning: Missing opacity for {base_prefix}, using zeros.")
            opacity = torch.zeros(1, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1])
            
        # 3. Depth
        if os.path.exists(depth_path):
            dep = Image.open(depth_path)
            depth = self.geo_transform(dep)
        else:
            print(f"Warning: Missing depth for {base_prefix}, using zeros.")
            depth = torch.zeros(1, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1])
            
        # 4. Sky Mask
        if os.path.exists(sky_path):
            sky = Image.open(sky_path).convert('RGB')
            sky_img = self.norm_transform(sky)
        else:
            print(f"Warning: Missing sky for {sat_id}, using black.")
            sky_img = torch.zeros(3, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1])
            sky_img = (sky_img - 0.5) / 0.5

        # 拼接 5通道输入
        input_5ch = torch.cat([init_rgb, opacity, depth], dim=0)

        return {
            'input_5ch': input_5ch,
            'sky_image': sky_img,
            'filename': f"{base_prefix}_gen.png" # 输出文件名保持和输入前缀一致
        }

# ==========================================
# 3. 辅助函数 (保持不变)
# ==========================================
@torch.no_grad()
def encode_latents(vae, image):
    latents = vae.encode(image).latent_dist.sample()
    latents = latents * 0.18215
    return latents

@torch.no_grad()
def decode_latents(vae, latents):
    latents = latents / 0.18215
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

# ==========================================
# 4. 主推理函数
# ==========================================
def run_inference():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE)
    print(f"Running inference on {device}")
    print(f"Mode: {'Sequence (Video)' if cfg.SEQ_MODE else 'Single Image'}")

    # ----------------------
    # A. 加载模型
    # ----------------------
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    print("Loading SatDiT...")
    model = SatDiT(
        input_size=(32, 64), 
        patch_size=2, 
        in_channels=4, 
        cond_channels=5,
        hidden_size=1024, 
        depth=28, 
        num_heads=16
    ).to(device)
    
    print(f"Loading weights from {cfg.CHECKPOINT_PATH}")
    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Scheduler
    scheduler = DDPMScheduler.from_pretrained("facebook/DiT-XL-2-256", subfolder="scheduler")
    scheduler.set_timesteps(cfg.NUM_INFERENCE_STEPS)

    # ----------------------
    # B. 数据准备 (传入 seq_mode)
    # ----------------------
    dataset = InferenceDataset(cfg.INPUT_DIR, cfg.SKY_DIR, seq_mode=cfg.SEQ_MODE)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Found {len(dataset)} frames to process.")

    # ----------------------
    # C. 推理循环
    # ----------------------
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_5ch = batch['input_5ch'].to(device)
            sky_img = batch['sky_image'].to(device)
            filenames = batch['filename']

            bs = input_5ch.shape[0]

            # 1. 准备条件 Latents
            input_rgb = input_5ch[:, :3, :, :]
            input_geo = input_5ch[:, 3:, :, :]

            cond_rgb_latents = encode_latents(vae, input_rgb)
            geo_latents = model.geo_encoder(input_geo)
            input_5ch_latents = torch.cat([cond_rgb_latents, geo_latents], dim=1)

            # 2. 初始化纯噪声
            latents = torch.randn(bs, 4, 32, 64).to(device)

            # 3. 扩散去噪循环
            for t in scheduler.timesteps:
                t = t.to(device)
                model_output = model(latents, t, sky_img, input_5ch_latents)
                
                if model.out_channels == 8:
                    model_output, _ = model_output.chunk(2, dim=1)
                
                latents = scheduler.step(model_output, t, latents).prev_sample

            # 4. 解码并保存
            images = decode_latents(vae, latents)

            for i in range(bs):
                save_path = os.path.join(cfg.OUTPUT_DIR, filenames[i])
                save_image(images[i], save_path)
                
    print(f"All done! Results saved to {cfg.OUTPUT_DIR}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2 SatDiT inference on Stage 1 frames.")
    parser.add_argument("--input_dir", type=str, default=cfg.INPUT_DIR, help="Folder containing *_rgb.png, *_depth.png, *_opacity.png.")
    parser.add_argument("--sky_dir", type=str, default=cfg.SKY_DIR, help="Folder containing sky masks/images named by satellite id.")
    parser.add_argument("--output_dir", type=str, default=cfg.OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=cfg.CHECKPOINT_PATH)
    parser.add_argument("--steps", type=int, default=cfg.NUM_INFERENCE_STEPS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--single_image", action="store_true", help="Use SatID_rgb.png naming instead of SatID_SeqID_rgb.png.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.INPUT_DIR = args.input_dir
    cfg.SKY_DIR = args.sky_dir
    cfg.OUTPUT_DIR = args.output_dir
    cfg.CHECKPOINT_PATH = args.checkpoint
    cfg.NUM_INFERENCE_STEPS = args.steps
    cfg.BATCH_SIZE = args.batch_size
    cfg.SEQ_MODE = not args.single_image
    run_inference()
