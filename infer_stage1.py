import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
import torchvision
from scipy.interpolate import splprep, splev

# 确保 model_s1.py 在同级目录
from model_s1 import SatToPanoModel, render, get_original_coord

# -----------------------------------------------------------------------------
# 1. 交互式画路径工具
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1. 交互式画路径工具 (修改版：支持固定步长采样)
# -----------------------------------------------------------------------------

def get_path_length(points):
    """计算路径总长度"""
    if len(points) < 2:
        return 0.0
    pts = np.array(points)
    # 计算相邻点之间的欧氏距离
    diffs = np.diff(pts, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(dists)

def draw_path_interactive(img_path, sat_size=(256, 256), step_size=0.02):
    """
    针对每一张图单独画路径
    :param step_size: 归一化坐标系下的采样步长 (默认0.02)。
                      范围通常是[-1, 1]，总宽2.0。
                      0.02 意味着跨越整张图大约需要 100 帧。
    """
    img_name = os.path.basename(img_path)
    print(f"\n>>> 正在为图片 [{img_name}] 绘制路径...")
    print("----------------------------")
    print("  [左键点击] 添加关键点")
    print("  [右键] 撤销")
    print("  [Enter/空格] 完成绘制并开始渲染")
    print("  [D 键] 使用默认直线路径 (跳过绘制)")
    print("----------------------------")

    # 读取并 Resize 图片用于显示
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((sat_size[0], sat_size[1], 3), np.uint8)
    img = cv2.resize(img, (sat_size[1], sat_size[0]))
    
    points = []
    temp_img = img.copy()
    window_name = f"Draw Path: {img_name}"

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_img, (x, y), 3, (0, 0, 255), -1)
            # 画线连接
            if len(points) > 1:
                cv2.line(temp_img, points[-2], points[-1], (0, 0, 255), 2)
            cv2.imshow(window_name, temp_img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(points) > 0:
                points.pop()
                temp_img[:] = img[:] # 重置背景
                # 重绘所有点和线
                for i, pt in enumerate(points):
                    cv2.circle(temp_img, pt, 3, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(temp_img, points[i-1], pt, (0, 0, 255), 2)
                cv2.imshow(window_name, temp_img)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, temp_img)

    use_default = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32: # Enter or Space
            break
        elif key == ord('d') or key == ord('D'):
            use_default = True
            print("  -> 已选择默认路径")
            break
    
    cv2.destroyAllWindows()

    if use_default or len(points) < 2:
        if not use_default: print("  -> 点数过少，自动使用默认路径。")
        return get_default_path(step_size=step_size)

    # 1. 坐标归一化
    H, W = sat_size
    normalized_path = []
    for (px, py) in points: 
        norm_w = (px / W) * 2 - 1 # Col -> W (X)
        norm_h = (py / H) * 2 - 1 # Row -> H (Y)
        normalized_path.append((norm_h, norm_w)) # Model requires [H, W]
    
    # 2. 根据步长动态计算帧数
    total_len = get_path_length(normalized_path)
    # 计算帧数 = 总长 / 步长，至少保留 10 帧
    num_frames = int(total_len / step_size)
    num_frames = max(10, num_frames)
    
    print(f"  -> 路径长度: {total_len:.2f}, 自动计算帧数: {num_frames} (步长: {step_size})")

    return interpolate_path(normalized_path, num_frames=num_frames)

def get_default_path(step_size=0.02):
    # 默认从下往上: (-0.8, 0) -> (0.8, 0)
    # 长度 = 1.6
    total_len = 1.6
    num_frames = int(total_len / step_size)
    
    t = np.linspace(-0.8, 0.8, num_frames)
    path = np.stack([t, np.zeros_like(t)], axis=1) # [H_vary, W_fixed]
    return path

def interpolate_path(key_points, num_frames):
    """
    使用 B 样条插值生成平滑路径
    """
    key_points = np.array(key_points)
    
    # 简单去重：如果相邻点距离过近则合并，防止 splprep 报错
    diff = np.diff(key_points, axis=0)
    dist = np.sum(diff ** 2, axis=1)
    valid_idx = np.concatenate(([True], dist > 1e-6))
    key_points = key_points[valid_idx]

    # 如果点太少，直接线性重采样
    if len(key_points) < 2: 
        return np.resize(key_points, (num_frames, 2))

    try:
        # B-spline 插值
        # k 是样条阶数，点很少时降低阶数
        k_val = 3 if len(key_points) > 3 else (len(key_points) - 1)
        tck, u = splprep(key_points.T, s=0, k=k_val)
        
        # 在 u 的范围内生成 num_frames 个点
        u_new = np.linspace(u.min(), u.max(), num_frames)
        h_new, w_new = splev(u_new, tck, der=0)
        path = np.stack([h_new, w_new], axis=1)
    except:
        # 如果样条插值失败（比如路径太乱），回退到基于距离的线性插值
        print("  [Warning] B-spline 插值失败，使用线性插值。")
        dists = np.sqrt(np.sum(np.diff(key_points, axis=0)**2, axis=1))
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        target = np.linspace(0, cum_dist[-1], num_frames)
        h = np.interp(target, cum_dist, key_points[:, 0])
        w = np.interp(target, cum_dist, key_points[:, 1])
        path = np.stack([h, w], axis=1)
        
    return path

def save_density_as_ply(density, save_path):
    if isinstance(density, torch.Tensor):
        density = density.detach().cpu().numpy()
    threshold = 0.1
    indices = np.argwhere(density > threshold)
    D, H, W = density.shape
    #header = f"ply\nformat ascii 1.0\nelement vertex {len(indices)}\nproperty float x\nproperty float y\nproperty float z\nproperty float density\nend_header\n"
    header = f"ply\nformat ascii 1.0\nelement vertex {len(indices)}\nproperty float x\nproperty float y\nproperty float z\nproperty float quality\nend_header\n"
    with open(save_path, 'w') as f:
        f.write(header)
        for (d, h, w) in indices:
            z, x, y = (d/D)*2-1, (h/H)*2-1, (w/W)*2-1
            f.write(f"{y:.4f} {x:.4f} {z:.4f} {density[d,h,w]:.4f}\n")

# -----------------------------------------------------------------------------
# 2. 主测试逻辑
# -----------------------------------------------------------------------------
def test(opt, args):
    os.makedirs(opt.output_dir, exist_ok=True)
    device = opt.device
    
    # 1. 初始化模型
    sam_ckpt = args.sam_checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = SatToPanoModel(opt, sam_ckpt).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    pano_direction = torch.from_numpy(get_original_coord(opt)).unsqueeze(0).to(device)

    # 2. 处理输入路径 (文件 or 文件夹)
    image_paths = []
    if os.path.isfile(args.input):
        image_paths = [args.input]
        print(f"模式: 单张图片测试 ({args.input})")
    elif os.path.isdir(args.input):
        files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_paths = [os.path.join(args.input, f) for f in files]
        print(f"模式: 文件夹测试 ({len(image_paths)} 张图片)")
    else:
        print(f"错误: 输入路径 {args.input} 不存在")
        return

    # 3. 循环处理
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 读取
        sat_img = cv2.imread(img_path)
        if sat_img is None: continue
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        sat_img = cv2.resize(sat_img, (opt.data.sat_size[1], opt.data.sat_size[0]))
        sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            # 推理 voxel
            density_voxel = model.density_generator(sat_tensor)
            sat_img_01 = sat_tensor / 255.0
            N_depth = model.density_channels + (1 if opt.optim.ground_prior else 0)
            color_volume = model.color_refiner(sat_img_01, N_depth, density_voxel)
            
            # 保存 PLY
            # save_density_as_ply(density_voxel[0, 0], os.path.join(opt.output_dir, f"{base_name}_density.ply"))
            save_density_as_ply(density_voxel[0], os.path.join(opt.output_dir, f"{base_name}_density.ply"))
            
            # 保存原始密度矩阵，用于本地体绘制
            np.save(os.path.join(opt.output_dir, f"{base_name}_density.npy"), density_voxel[0].cpu().numpy())
            
            # ---------------------------------------------------------
            # 渲染部分 (修改了这里)
            # ---------------------------------------------------------
            if not args.move:
                # === 静态单张 ===
                opt.origin_H_W = None 
                curr_dir = pano_direction.repeat(1, 1, 1, 1)
                out = render(opt, color_volume, density_voxel, curr_dir)
                
                # 1. 保存 RGB
                torchvision.utils.save_image(out['rgb'], os.path.join(opt.output_dir, f"{base_name}_rgb.png"))
                
                # 2. 保存 Opacity (兼容常见键名: 'opacity', 'acc', 'weights_sum')
                # 优先找 'opacity'，找不到找 'acc'，再找不到找 'weights_sum'
                opacity_map = out.get('opacity', out.get('acc', out.get('weights_sum', None)))
                if opacity_map is not None:
                    torchvision.utils.save_image(opacity_map, os.path.join(opt.output_dir, f"{base_name}_opacity.png"))
                
                # 3. 保存 Depth
                if 'depth' in out:
                    # 注意：如果深度值非常小或非常大，直接保存可能全黑或全白。
                    # 这里直接保存原始值，如果需要可视化更清晰，可以添加 normalize=True 参数
                    torchvision.utils.save_image(out['depth'], os.path.join(opt.output_dir, f"{base_name}_depth.png"))
                
                print(f"[Done] {base_name} saved (rgb/opacity/depth).")
                
            else:
                # === 动态漫游 ===
                current_path = draw_path_interactive(img_path, opt.data.sat_size)
                
                print(f"Rendering {len(current_path)} frames for {base_name}...")
                for i, (nh, nw) in enumerate(current_path):
                    opt.origin_H_W = [nh, nw]
                    out = render(opt, color_volume, density_voxel, pano_direction)
                    
                    idx_str = str(i).zfill(3)
                    
                    # 1. 保存 RGB
                    save_name_rgb = os.path.join(opt.output_dir, f"{base_name}_{idx_str}_rgb.png")
                    torchvision.utils.save_image(out['rgb'], save_name_rgb)
                    
                    # 2. 保存 Opacity
                    opacity_map = out.get('opacity', out.get('acc', out.get('weights_sum', None)))
                    if opacity_map is not None:
                        save_name_op = os.path.join(opt.output_dir, f"{base_name}_{idx_str}_opacity.png")
                        torchvision.utils.save_image(opacity_map, save_name_op)
                        
                    # 3. 保存 Depth
                    if 'depth' in out:
                        save_name_depth = os.path.join(opt.output_dir, f"{base_name}_{idx_str}_depth.png")
                        torchvision.utils.save_image(out['depth'], save_name_depth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage1/best_model.pth", help="模型权重路径")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_l_0b3195.pth", help="SAM checkpoint path kept for compatibility with the model constructor.")
    parser.add_argument("--input", type=str, default="data/CVUSA_train/train/sat_images", help="单张图片路径 或 文件夹路径")
    parser.add_argument("--move", action='store_true', help="开启路径绘制漫游模式")
    parser.add_argument("--output_dir", type=str, default="CVUSA_results-full")
    
    args = parser.parse_args()
    
    # Config
    opt = edict()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.data = edict()
    #opt.data.dataset = 'CVACT_Shi' # 必须与训练一致
    opt.data.dataset = 'CVUSA' # 必须与训练一致
    opt.data.sat_size = [256, 256]
    opt.data.pano_size = [512, 256]
    opt.data.max_height = 15
    opt.data.sample_number = 256
    opt.data.sample_total_length = 1.0
    opt.arch = edict()
    opt.arch.gen = edict()
    opt.arch.gen.depth_arch = edict()
    opt.arch.gen.depth_arch.output_nc = 65
    opt.arch.gen.PE_channel = 2
    opt.mlp_hidden_dim = 64
    opt.optim = edict()
    opt.optim.ground_prior = True
    opt.origin_H_W = None 
    opt.output_dir = args.output_dir

    test(opt, args)
