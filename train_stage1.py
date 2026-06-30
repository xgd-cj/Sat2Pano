# 文件名: train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
from tqdm import tqdm
import torchvision
from PIL import Image
import shutil
import torch.nn.functional as F
import argparse 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler 
import torch.multiprocessing as mp # [新增] 用于 mp.spawn

# 导入自己的模块
from model_s1 import SatToPanoModel
from dataset_s1 import get_loaders 

# -------------------------------------------------------------------
# 卫星渲染器逻辑 
# -------------------------------------------------------------------

def composite_sat(opt, rgb_samples, density_samples, depth_samples, intv):
    sigma_delta = density_samples * intv 
    sigma_delta = F.relu(sigma_delta) 
    
    alpha = 1 - (-sigma_delta).exp_() 
    T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=-1).cumsum(dim=-1)).exp_() 
    prob = (T * alpha)[..., None] 
    
    rgb = (rgb_samples * prob).sum(dim=2) 
    opacity = prob.sum(dim=2) 
    
    H_sat, W_sat = opt.data.sat_size
    
    rgb = rgb.permute(0, 2, 1).view(rgb.size(0), -1, H_sat, W_sat)
    opacity = opacity.permute(0, 2, 1).view(opacity.size(0), 1, H_sat, W_sat)
    
    depth = (depth_samples * prob).sum(dim=2) 
    depth = depth.permute(0, 2, 1).view(depth.size(0), -1, H_sat, W_sat)

    return {'rgb': rgb, 'opacity': opacity, 'depth': depth}


class SatRenderer(nn.Module):
    def __init__(self, opt: edict):
        super().__init__()
        self.opt = opt
        if opt.data.dataset =='CVACT_Shi':
            realworld_scale = 30
        elif opt.data.dataset == 'CVUSA':
            realworld_scale = 55
        else:
            realworld_scale = 30 
        
        total_norm_height = opt.data.max_height / (realworld_scale / 2.0)
        N_base = opt.arch.gen.depth_arch.output_nc
        N_total = N_base + 1 if opt.optim.ground_prior else N_base
        self.intv_z = total_norm_height / N_total

    def forward(self, color_volume: torch.Tensor, density_voxel: torch.Tensor):
        B, C_color, N_color, H, W = color_volume.shape
        _, N_base, _, _ = density_voxel.shape
        
        if self.opt.optim.ground_prior:
            ground_plane = torch.ones(B, 1, H, W, device=density_voxel.device) * 1000.0
            density_volume = torch.cat([ground_plane, density_voxel], 1)
        else:
            density_volume = density_voxel
        
        assert N_color == density_volume.shape[1], \
            f"Color depth ({N_color}) and Density depth ({density_volume.shape[1]}) must match."
        
        N = N_color
        rgb_samples = torch.flip(color_volume, dims=[2]).permute(0, 3, 4, 2, 1).reshape(B, H*W, N, C_color)
        density_samples = torch.flip(density_volume, dims=[2]).permute(0, 2, 3, 1).reshape(B, H*W, N)
        depth_samples = torch.zeros(B, H*W, N, 1, device=density_volume.device)
        
        output = composite_sat(
            self.opt,
            rgb_samples=rgb_samples,
            density_samples=density_samples,
            depth_samples=depth_samples,
            intv=self.intv_z
        )
        return output


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

# DDP 辅助函数：汇聚所有卡上的 Loss
def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

# DDP 辅助函数：判断是否为主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_config(args=None) -> edict:
    opt = edict()

    # --- 路径配置 ---
    opt.data = edict()
    opt.data.root_dir = 'CVUSA_train' 
    opt.data.train_split = 'train' 
    opt.data.val_split = 'val' 
    
    opt.train = edict()
    opt.train.sam_checkpoint_path = "sam_vit_l_0b3195.pth"
    opt.train.checkpoint_dir = "checkpointsS1_CVUSA"
    opt.train.log_dir = "logs" 
    opt.train.output_dir = "train_outputs_CVUSA"
    
    # 断点续训路径
    opt.train.resume_path = args.resume if args and args.resume else None 

    # --- 训练参数 ---
    opt.train.epochs = 150
    opt.train.batch_size = 2 # DDP模式下这是单卡 Batch Size
    opt.train.num_workers = 2
    opt.train.lr = 1e-4
    opt.train.lr_decay_step = 20
    opt.train.lr_decay_gamma = 0.5
    opt.train.log_interval = 10
    opt.train.val_interval = 1
    
    opt.train.loss_weight_l1 = 1.0 
    opt.train.loss_weight_sky = 0.05 
    opt.train.loss_weight_sat_recon = 0 

    # --- 模型配置 ---
    opt.device = 'cuda' 

    # 数据参数
    opt.data.pano_size = [512, 256] 
    opt.data.sat_size = [256, 256]  
    opt.data.dataset = 'CVUSA'
    opt.data.max_height = 15
    opt.data.sample_number = 256
    opt.data.sample_total_length = 1.0

    # 架构参数
    opt.arch = edict()
    opt.arch.gen = edict()
    opt.arch.gen.depth_arch = edict()
    opt.arch.gen.depth_arch.output_nc = 65
    opt.arch.gen.PE_channel = 2
    opt.mlp_hidden_dim = 64

    opt.optim = edict()
    opt.optim.ground_prior = True
    opt.origin_H_W = None 

    # 将命令行参数存入 opt
    if args:
        if args.data_root:
            opt.data.root_dir = args.data_root
        if args.train_split:
            opt.data.train_split = args.train_split
        if args.val_split:
            opt.data.val_split = args.val_split
        if args.sam_checkpoint:
            opt.train.sam_checkpoint_path = args.sam_checkpoint
        if args.checkpoint_dir:
            opt.train.checkpoint_dir = args.checkpoint_dir
        if args.log_dir:
            opt.train.log_dir = args.log_dir
        if args.output_dir:
            opt.train.output_dir = args.output_dir
        if args.epochs is not None:
            opt.train.epochs = args.epochs
        if args.batch_size is not None:
            opt.train.batch_size = args.batch_size
        if args.num_workers is not None:
            opt.train.num_workers = args.num_workers
        if args.lr is not None:
            opt.train.lr = args.lr
        opt.distributed = args.distributed
        opt.local_rank = args.local_rank
    else:
        opt.distributed = False
        opt.local_rank = 0

    return opt

def setup_dummy_data(opt: edict):
    # DDP 模式下，只有主进程检查/创建数据
    if not is_main_process():
        dist.barrier() # 等待主进程
        return

    train_sat_dir = os.path.join(opt.data.root_dir, opt.data.train_split, 'sat_images')
    if os.path.exists(train_sat_dir) and len(os.listdir(train_sat_dir)) > 0:
        if opt.distributed: dist.barrier()
        return

    # 这里省略创建 dummy data 的具体代码，假设数据已存在或你会自己处理
    print("注意：setup_dummy_data 被调用，请确保数据集路径正确。")
    
    if opt.distributed: dist.barrier() 

def validate(model: SatToPanoModel, sat_renderer: nn.Module, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, opt: edict, epoch: int, writer: SummaryWriter):
    model.eval() 
    total_loss = 0.0
    total_l1_loss = 0.0 
    total_sky_loss = 0.0 
    total_sat_recon_loss = 0.0 
    total_psnr = 0.0 
    
    if is_main_process():
        epoch_output_dir = os.path.join(opt.train.output_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_output_dir, exist_ok=True)
    
    if is_main_process():
        print(f"\n--- 开始验证 Epoch {epoch} ---")
        pbar = tqdm(val_loader, desc=f"验证 Epoch {epoch}")
    else:
        pbar = val_loader

    pano_comparison_grid = None
    sat_comparison_grid = None
    
    with torch.no_grad():
        for i, (sat_img, pano_img, sky_mask) in enumerate(pbar):
            sat_img = sat_img.to(opt.device)
            pano_img = pano_img.to(opt.device)
            sky_mask = sky_mask.to(opt.device) 
            
            model_outputs = model(sat_img)
            
            pano_render = model_outputs['pano_render']
            pred_rgb_pano = pano_render['rgb']
            pred_opacity_pano = pano_render['opacity']
            density_voxel = model_outputs['density_voxel']
            color_volume = model_outputs['color_volume']
            sat_render = sat_renderer(color_volume, density_voxel)
            pred_rgb_sat = sat_render['rgb'] 

            loss_l1 = criterion(pred_rgb_pano, pano_img)
            sky_target = 1.0 - sky_mask 
            loss_sky = criterion(pred_opacity_pano, sky_target)
            sat_img_01 = sat_img / 255.0 
            loss_sat_recon = criterion(pred_rgb_sat, sat_img_01)
            
            loss = (opt.train.loss_weight_l1 * loss_l1) + \
                   (opt.train.loss_weight_sky * loss_sky) + \
                   (opt.train.loss_weight_sat_recon * loss_sat_recon) 
            
            total_loss += loss.item()
            total_l1_loss += loss_l1.item() 
            total_sky_loss += loss_sky.item() 
            total_sat_recon_loss += loss_sat_recon.item() 
            
            psnr = calculate_psnr(pred_rgb_pano, pano_img)
            total_psnr += psnr
            
            if is_main_process():
                pbar.set_postfix({"总损失": loss.item(), "PSNR": f"{psnr:.2f} dB"})

                if i == 0: 
                    B = pred_rgb_pano.size(0)
                    opacity_map = pred_opacity_pano.repeat(1, 3, 1, 1)
                    sky_target_viz = sky_target.repeat(1, 3, 1, 1)
                    
                    pano_comparison_grid = torch.cat([pano_img, pred_rgb_pano, opacity_map, sky_target_viz], dim=3)
                    sat_comparison_grid = torch.cat([sat_img_01, pred_rgb_sat], dim=3)
                    
                    save_path = os.path.join(epoch_output_dir, f"val_pano_batch_0.png")
                    torchvision.utils.save_image(pano_comparison_grid, save_path, nrow=B)
                    save_path_sat = os.path.join(epoch_output_dir, f"val_sat_batch_0.png")
                    torchvision.utils.save_image(sat_comparison_grid, save_path_sat, nrow=B)

    if opt.distributed:
        avg_loss = reduce_value(torch.tensor(total_loss / len(val_loader), device=opt.device)).item()
    else:
        avg_loss = total_loss / len(val_loader)
        
    avg_l1_loss = total_l1_loss / len(val_loader) 
    avg_sky_loss = total_sky_loss / len(val_loader) 
    avg_sat_recon_loss = total_sat_recon_loss / len(val_loader) 
    avg_psnr = total_psnr / len(val_loader)
    
    if is_main_process():
        print(f"--- 验证 Epoch {epoch} 完成 ---")
        print(f"平均总损失: {avg_loss:.6f}")
        print(f"  (L1: {avg_l1_loss:.6f}, Sky: {avg_sky_loss:.6f}, SatRecon: {avg_sat_recon_loss:.6f})") 
        print(f"平均 PSNR: {avg_psnr:.4f} dB")
        
        writer.add_scalar('Metrics/Validation_Total_Loss', avg_loss, epoch)
        writer.add_scalar('Metrics/Validation_L1_Loss', avg_l1_loss, epoch)
        writer.add_scalar('Metrics/Validation_Sky_Loss', avg_sky_loss, epoch)
        writer.add_scalar('Metrics/Validation_Sat_Recon_Loss', avg_sat_recon_loss, epoch) 
        writer.add_scalar('Metrics/Validation_PSNR', avg_psnr, epoch)
        
        if pano_comparison_grid is not None:
            writer.add_images('Validation/Pano_Comparison (GT, Pred, Opacity, SkyTarget)', pano_comparison_grid, epoch)
        if sat_comparison_grid is not None: 
            writer.add_images('Validation/Sat_Comparison (GT, Pred)', sat_comparison_grid, epoch)
    
    return avg_loss

def main(opt: edict):
    # 设置当前进程的 device
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        opt.device = torch.device('cuda', opt.local_rank)
    else:
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_main_process():
        print("开始训练...")
        print(f"使用设备: {opt.device} (Distributed: {opt.distributed})")
        os.makedirs(opt.train.checkpoint_dir, exist_ok=True)
        os.makedirs(opt.train.log_dir, exist_ok=True)
        os.makedirs(opt.train.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=opt.train.log_dir)
    else:
        writer = None
    
    setup_dummy_data(opt) 

    if is_main_process(): print("正在加载数据...")
    
    # 获取 DataLoader
    train_loader_orig, val_loader_orig = get_loaders(opt)
    
    if opt.distributed:
        # 提取 Dataset 并重新包装 Sampler
        train_dataset = train_loader_orig.dataset
        val_dataset = val_loader_orig.dataset
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=opt.train.batch_size, 
            shuffle=False, 
            num_workers=opt.train.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=opt.train.batch_size, 
            shuffle=False,
            num_workers=opt.train.num_workers,
            pin_memory=True,
            sampler=val_sampler
        )
    else:
        train_loader = train_loader_orig
        val_loader = val_loader_orig
        train_sampler = None

    if is_main_process(): print("正在初始化模型...")
        
    model = SatToPanoModel(opt, opt.train.sam_checkpoint_path).to(opt.device)
    sat_renderer = SatRenderer(opt).to(opt.device)

    optimizer = optim.Adam(model.parameters(), lr=opt.train.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=opt.train.lr_decay_step, 
        gamma=opt.train.lr_decay_gamma
    )

    # -------------------------------------------------------------------------
    # 断点续训 (Resume)
    # -------------------------------------------------------------------------
    start_epoch = 1
    best_val_loss = float('inf')

    if opt.train.resume_path and os.path.exists(opt.train.resume_path):
        if is_main_process():
            print(f"检测到 Checkpoint，正在从 {opt.train.resume_path} 恢复训练...")
        
        checkpoint = torch.load(opt.train.resume_path, map_location=opt.device)
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            
        if is_main_process():
            print(f"成功恢复! 将从 Epoch {start_epoch} 开始继续训练。")
    else:
        if opt.train.resume_path and is_main_process():
             print(f"警告: 指定的 resume_path {opt.train.resume_path} 不存在，将从头开始训练。")

    # DDP 包装
    if opt.distributed:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)

    if is_main_process():
        model_params = model.module if opt.distributed else model
        total_params = sum(p.numel() for p in model_params.parameters() if p.requires_grad)
        print(f"模型准备就绪。总可训练参数: {total_params / 1e6:.2f} M")

    criterion = nn.L1Loss() 
    
    for epoch in range(start_epoch, opt.train.epochs + 1):
        
        if opt.distributed:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n===== Epoch {epoch}/{opt.train.epochs} ===== LR: {scheduler.get_last_lr()[0]:.1e}")
        
        model.train() 
        train_loss_epoch = 0.0
        train_l1_loss_epoch = 0.0 
        train_sky_loss_epoch = 0.0 
        train_sat_recon_loss_epoch = 0.0 
        train_psnr_epoch = 0.0
        
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
        else:
            pbar = train_loader

        for i, (sat_img, pano_img, sky_mask) in enumerate(pbar):
            sat_img = sat_img.to(opt.device)
            pano_img = pano_img.to(opt.device)
            sky_mask = sky_mask.to(opt.device) 
            
            model_outputs = model(sat_img)
            
            pano_render = model_outputs['pano_render']
            density_voxel = model_outputs['density_voxel']
            color_volume = model_outputs['color_volume']
            
            pred_rgb_pano = pano_render['rgb']
            pred_opacity_pano = pano_render['opacity']
            
            sat_render = sat_renderer(color_volume, density_voxel)
            pred_rgb_sat = sat_render['rgb']
            
            loss_l1 = criterion(pred_rgb_pano, pano_img)
            sky_target = 1.0 - sky_mask
            loss_sky = criterion(pred_opacity_pano, sky_target)
            sat_img_01 = sat_img / 255.0 
            loss_sat_recon = criterion(pred_rgb_sat, sat_img_01) 
            
            loss = (opt.train.loss_weight_l1 * loss_l1) + \
                   (opt.train.loss_weight_sky * loss_sky) + \
                   (opt.train.loss_weight_sat_recon * loss_sat_recon) 
            
            with torch.no_grad():
                psnr = calculate_psnr(pred_rgb_pano, pano_img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_l1_loss_epoch += loss_l1.item()
            train_sky_loss_epoch += loss_sky.item()
            train_sat_recon_loss_epoch += loss_sat_recon.item() 
            train_psnr_epoch += psnr
            
            if is_main_process() and (i + 1) % opt.train.log_interval == 0:
                current_loss = train_loss_epoch / (i + 1)
                current_psnr = train_psnr_epoch / (i + 1)
                pbar.set_postfix({"总损失": f"{current_loss:.6f}", "PSNR": f"{current_psnr:.2f} dB"})
                
                global_step = (epoch - 1) * len(train_loader) + i
                writer.add_scalar('Loss/Train_Batch_Total', loss.item(), global_step)
                writer.add_scalar('Loss/Train_Batch_L1', loss_l1.item(), global_step)
                writer.add_scalar('Loss/Train_Batch_Sky', loss_sky.item(), global_step)
                writer.add_scalar('Loss/Train_Batch_Sat_Recon', loss_sat_recon.item(), global_step) 
                writer.add_scalar('Metrics/Train_Batch_PSNR', psnr, global_step)

        if opt.distributed:
            avg_train_loss = reduce_value(torch.tensor(train_loss_epoch / len(train_loader), device=opt.device)).item()
        else:
            avg_train_loss = train_loss_epoch / len(train_loader)

        avg_train_l1_loss = train_l1_loss_epoch / len(train_loader)
        avg_train_sky_loss = train_sky_loss_epoch / len(train_loader)
        avg_train_sat_recon_loss = train_sat_recon_loss_epoch / len(train_loader) 
        avg_train_psnr = train_psnr_epoch / len(train_loader)
        
        if is_main_process():
            writer.add_scalar('Metrics/Train_Epoch_Total_Loss', avg_train_loss, epoch)
            writer.add_scalar('Metrics/Train_Epoch_L1_Loss', avg_train_l1_loss, epoch)
            writer.add_scalar('Metrics/Train_Epoch_Sky_Loss', avg_train_sky_loss, epoch)
            writer.add_scalar('Metrics/Train_Epoch_Sat_Recon_Loss', avg_train_sat_recon_loss, epoch) 
            writer.add_scalar('Metrics/Train_Epoch_PSNR', avg_train_psnr, epoch)
            
            pbar.close()
            print(f"Epoch {epoch} 训练完成。平均总损失: {avg_train_loss:.6f}")
            print(f"  (L1: {avg_train_l1_loss:.6f}, Sky: {avg_train_sky_loss:.6f}, SatRecon: {avg_train_sat_recon_loss:.6f}, PSNR: {avg_train_psnr:.4f} dB)") 
        
        scheduler.step()

        if epoch % opt.train.val_interval == 0:
            avg_val_loss = validate(model, sat_renderer, val_loader, criterion, opt, epoch, writer)
            
            if is_main_process():
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if opt.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }
                
                latest_save_path = os.path.join(opt.train.checkpoint_dir, "latest_model.pth")
                torch.save(checkpoint_dict, latest_save_path)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    checkpoint_dict['best_val_loss'] = best_val_loss
                    
                    save_path = os.path.join(opt.train.checkpoint_dir, "best_model.pth")
                    torch.save(checkpoint_dict, save_path)
                    print(f"新最佳模型! 验证总损失: {avg_val_loss:.6f}。已保存到 {save_path}")

            
    if is_main_process():
        print("\n--- 训练完成 ---")
        writer.close()
        if opt.data.root_dir == 'dummy_dataset':
            print(f"清理虚拟数据集: {opt.data.root_dir}")
            shutil.rmtree(opt.data.root_dir)

# -------------------------------------------------------------------
# [新增] Worker 函数，用于 mp.spawn 调用
# -------------------------------------------------------------------
def main_worker(rank, world_size, args):
    """
    DDP 子进程入口
    """
    # 1. 设置通信环境 (localhost)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 

    # 2. 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # 3. 修正 args (local_rank 对应当前进程 rank)
    args.local_rank = rank
    args.distributed = True

    # 4. 获取配置并运行 main
    config = get_config(args)
    
    try:
        main(config)
    finally:
        dist.destroy_process_group()

# -------------------------------------------------------------------
# 程序入口
# -------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root with train/val subfolders.")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    # 允许手动指定 GPU 数量，若不指定则自动检测
    parser.add_argument("--n_gpus", type=int, default=None, help="Number of GPUs to use")
    
    args = parser.parse_args()

    # 1. 检测 GPU 数量
    n_gpus = torch.cuda.device_count()
    if args.n_gpus is not None:
        n_gpus = min(args.n_gpus, n_gpus)

    # 2. 根据 GPU 数量决定启动方式
    if n_gpus > 1:
        print(f"检测到 {n_gpus} 张显卡，正在启动多进程分布式训练 (mp.spawn)...")
        # 启动 n_gpus 个进程，每个进程执行 main_worker
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        print("启动单卡训练模式...")
        args.distributed = False
        args.local_rank = 0
        config = get_config(args)
        main(config)
