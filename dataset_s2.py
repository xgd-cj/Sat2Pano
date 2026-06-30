import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SatPanoramaDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        """
        data_root: 数据集根目录，下面应该包含:
            - satellite/  (卫星图)
            - ground_truth/ (真值全景图)
            - init_proj/  (初始投影图)
            - depth/      (深度图)
            - opacity/    (不透明度图)
            - sky/        (天空掩码图 - 如果没有，请看下文注释)
        """
        super().__init__()
        self.data_root = data_root
        
        # 定义各子文件夹名称 (请根据你实际的文件夹名修改这里)
        self.dirs = {
            'sat': os.path.join(data_root, 'satellite'),
            'gt': os.path.join(data_root, 'ground_truth'),
            'proj': os.path.join(data_root, 'init_proj'),
            'depth': os.path.join(data_root, 'depth'),
            'opacity': os.path.join(data_root, 'opacity'),
            'sky': os.path.join(data_root, 'sky') # 假设有这个文件夹
        }

        # 获取文件列表 (假设所有文件夹下的文件名都是一一对应的，以真值图目录为准)
        self.filenames = [f for f in os.listdir(self.dirs['gt']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 排序确保索引一致
        self.filenames.sort()

        # ==== 预处理变换 ====
        
        # 1. 卫星图变换: 256x256, 归一化到 [-1, 1]
        self.sat_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 2. 全景图/投影/天空变换: 256x512, 归一化到 [-1, 1]
        self.pano_transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 3. 几何信息(深度/不透明度)变换: 256x512, 单通道, 保持 [0, 1] 或归一化
        # 这里建议保持 [0, 1] 或 [-1, 1] 需与训练时的拼接保持一致
        # 通常作为 Condition，归一化到 [-1, 1] 会更稳定
        self.geo_transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.Grayscale(num_output_channels=1), # 强制转单通道
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # [0, 1] -> [-1, 1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # 辅助加载函数
        def load_img(folder_key, convert_mode='RGB'):
            path = os.path.join(self.dirs[folder_key], filename)
            return Image.open(path).convert(convert_mode)

        # 1. 加载并变换 RGB 图像
        gt_img = self.pano_transform(load_img('gt'))
        sat_img = self.sat_transform(load_img('sat'))
        proj_img = self.pano_transform(load_img('proj'))
        
        # 2. 加载并变换 几何图像 (单通道)
        depth_img = self.geo_transform(load_img('depth', 'L'))
        opacity_img = self.geo_transform(load_img('opacity', 'L'))

        # 3. 加载天空图
        # 如果你没有单独的天空文件夹，可以使用 proj_img 进行 mask 处理生成
        # 这里假设你有文件夹
        if os.path.exists(os.path.join(self.dirs['sky'], filename)):
            sky_img = self.pano_transform(load_img('sky'))
        else:
            # 【备用方案】如果没有天空图文件，临时生成一个全黑图
            # 建议你最好准备好天空图数据
            sky_img = torch.zeros_like(gt_img) 

        return {
            'gt_image': gt_img,
            'sat_image': sat_img,
            'cond_proj': proj_img,
            'cond_depth': depth_img,
            'cond_opacity': opacity_img,
            'sky_image': sky_img,
            'filename': filename # 返回文件名方便调试
        }