import os
import glob
import argparse
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from cleanfid import fid
from PIL import Image
from tqdm import tqdm

def read_img_for_lpips(path):
    """读取图像并转换为 LPIPS 需要的 Tensor 格式 (Range: [-1, 1])"""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0  # 0~1
    img = img * 2.0 - 1.0  # -1~1
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
    return img

def read_img_for_skimage(path):
    """读取图像并转换为 SSIM/PSNR 需要的 Numpy 格式 (Range: [0, 255], uint8)"""
    img = Image.open(path).convert('RGB')
    return np.array(img)

def evaluate_metrics(gen_dir, gt_dir, device='cuda'):
    """
    计算 SSIM, PSNR, LPIPS, FID, KID
    """
    print(f"正在评估文件夹:\n - 生成图像: {gen_dir}\n - 真值图像: {gt_dir}")
    
    # 检查设备
    if not torch.cuda.is_available() and device == 'cuda':
        print("警告: CUDA 不可用，切换到 CPU 模式（速度会变慢）。")
        device = 'cpu'

    # -------------------------------------------------------------------------
    # 1. 计算分布指标 (Distribution Metrics): FID, KID
    # -------------------------------------------------------------------------
    # FID/KID 是基于整个文件夹分布的，不需要文件名一一对应，只要两个文件夹代表两个分布即可
    print("\n[1/3] 正在计算 FID 和 KID (这可能需要下载 Inception 模型)...")
    
    # 这里的 num_workers=0 是为了防止之前出现的 Numpy 版本兼容性报错
    fid_score = fid.compute_fid(gen_dir, gt_dir, device=torch.device(device), num_workers=0)
    kid_score = fid.compute_kid(gen_dir, gt_dir, device=torch.device(device), num_workers=0)
    
    print(f" -> FID: {fid_score:.4f}")
    print(f" -> KID: {kid_score:.6f}")

    # -------------------------------------------------------------------------
    # 2. 准备成对数据 (Paired Data) - 修改了这里！
    # -------------------------------------------------------------------------
    print("\n[2/3] 正在进行文件配对 (处理 '_gen' 后缀)...")
    
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    # 获取所有生成图像路径
    gen_candidates = sorted([f for ext in exts for f in glob.glob(os.path.join(gen_dir, ext))])
    
    gen_files = []
    gt_files = []

    for g_path in gen_candidates:
        # 获取文件名，例如: "000123_gen.png"
        basename = os.path.basename(g_path)
        name, ext = os.path.splitext(basename) # name="000123_gen", ext=".png"
        
        # 去掉 "_gen" 后缀逻辑
        if name.endswith('_gen'):
            gt_name_base = name[:-4] # 去掉最后4个字符 -> "000123"
        else:
            gt_name_base = name      # 如果没有 _gen，保持原样

        # 构造预期的真值文件名
        gt_name = gt_name_base + ext
        gt_path = os.path.join(gt_dir, gt_name)

        # 只有当对应的真值文件存在时，才算配对成功
        if os.path.exists(gt_path):
            gen_files.append(g_path)
            gt_files.append(gt_path)
        else:
            # 如果没找到，尝试一下其他后缀 (比如生成是 png，真值是 jpg)
            found_alt = False
            for alt_ext in ['.png', '.jpg', '.jpeg']:
                gt_path_alt = os.path.join(gt_dir, gt_name_base + alt_ext)
                if os.path.exists(gt_path_alt):
                    gen_files.append(g_path)
                    gt_files.append(gt_path_alt)
                    found_alt = True
                    break
            # if not found_alt:
            #    print(f"跳过: 找不到对应的真值文件 -> {basename}")

    print(f" -> 成功配对 {len(gen_files)} 组图像。")

    if len(gen_files) == 0:
        print("错误: 没有找到任何匹配的图像对！请检查文件名后缀是否正确。")
        return

    # -------------------------------------------------------------------------
    # 3. 计算成对指标 (Paired Metrics): SSIM, PSNR, LPIPS
    # -------------------------------------------------------------------------
    print(f"正在初始化 LPIPS 模型 (net='alex')...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    ssim_scores = []
    psnr_scores = []
    lpips_scores = []

    print("[3/3] 正在逐张计算 SSIM, PSNR, LPIPS...")
    for gen_path, gt_path in tqdm(zip(gen_files, gt_files), total=len(gen_files)):
        
        # --- 计算 LPIPS (Tensor, GPU) ---
        t_gen = read_img_for_lpips(gen_path).to(device)
        t_gt = read_img_for_lpips(gt_path).to(device)
        
        with torch.no_grad():
            d = loss_fn_alex(t_gen, t_gt)
            lpips_scores.append(d.item())

        # --- 计算 SSIM & PSNR (Numpy, CPU) ---
        img_gen = read_img_for_skimage(gen_path)
        img_gt = read_img_for_skimage(gt_path)

        # 确保尺寸一致 (有些GAN生成的图可能差几个像素，这里做一个简单的resize保护)
        if img_gen.shape != img_gt.shape:
            # 如果尺寸不一致，通常把 GT resize 成生成的尺寸
            img_gt_pil = Image.fromarray(img_gt).resize((img_gen.shape[1], img_gen.shape[0]), Image.BICUBIC)
            img_gt = np.array(img_gt_pil)

        # PSNR
        p = psnr_func(img_gt, img_gen, data_range=255)
        psnr_scores.append(p)

        # SSIM
        s = ssim_func(img_gt, img_gen, data_range=255, channel_axis=-1) 
        ssim_scores.append(s)

    # -------------------------------------------------------------------------
    # 4. 汇总输出
    # -------------------------------------------------------------------------
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_lpips = np.mean(lpips_scores)

    print("\n" + "="*40)
    print("      全面评估结果 (Comprehensive Evaluation)      ")
    print("="*40)
    print(f"Low-Level Metrics (结构与像素保真度):")
    print(f"  SSIM  ↑ (越高越好): {avg_ssim:.4f}")
    print(f"  PSNR  ↑ (越高越好): {avg_psnr:.4f} dB")
    print("-" * 40)
    print(f"Perceptual Metrics (感知质量):")
    print(f"  LPIPS ↓ (越低越好): {avg_lpips:.4f}")
    print("-" * 40)
    print(f"Distribution Metrics (分布真实感):")
    print(f"  FID   ↓ (越低越好): {fid_score:.4f}")
    print(f"  KID   ↓ (越低越好): {kid_score:.6f}")
    print("="*40)

    return {
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
        "LPIPS": avg_lpips,
        "FID": fid_score,
        "KID": kid_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated panoramas against ground truth images.")
    parser.add_argument("--gen_dir", type=str, default="outputs/stage2_inference")
    parser.add_argument("--gt_dir", type=str, default="data/ground_truth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # 运行评估
    if os.path.exists(args.gen_dir) and os.path.exists(args.gt_dir):
        evaluate_metrics(args.gen_dir, args.gt_dir, device=args.device)
    else:
        print("错误：请输入正确的文件夹路径。")
