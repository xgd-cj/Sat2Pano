# 文件名: dataloader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from easydict import EasyDict as edict

def scale_to_255(tensor):
    return tensor * 255.0


class SatPanoDataset(Dataset):
    """
    自定义数据集，用于加载 (卫星图, 全景图, 天空掩码) 对。
    
    它假设数据结构如下:
    - {root_dir}/{data_split}/sat_images/
    - {root_dir}/{data_split}/pano_images/
    - {root_dir}/{data_split}/sky_masks/  <--- [新增]
    
    它将自动匹配三个文件夹中具有相同基本文件名 (不含扩展名) 的图像。
    """
    def __init__(self, opt: edict, data_split: str, is_train: bool):
        """
        Args:
            opt (edict): 配置对象, 包含 data.root_dir, data.sat_size, data.pano_size。
            data_split (str): 数据划分 (例如 'train' 或 'val')。
            is_train (bool): 是否为训练集 (此参数在此版本中被忽略)。
        """
        self.opt = opt
        self.root_dir = opt.data.root_dir
        
        # 定义数据子目录
        self.data_split_dir = os.path.join(self.root_dir, data_split)
        self.sat_dir = os.path.join(self.data_split_dir, 'sat_images')
        self.pano_dir = os.path.join(self.data_split_dir, 'pano_images')
        self.sky_dir = os.path.join(self.data_split_dir, 'sky_masks') # <--- [新增] 天空掩码目录
        
        # 目标尺寸
        self.sat_size = (opt.data.sat_size[0], opt.data.sat_size[1]) # (H, W)
        self.pano_size = (opt.data.pano_size[1], opt.data.pano_size[0]) # (H, W)
        
        # 自动查找图像对 (现在是三元组)
        self.image_triplets = self._find_image_triplets() # <--- [修改]
        
        if not self.image_triplets:
            raise FileNotFoundError(f"在 {self.data_split_dir} 中未找到匹配的图像三元组。")
        
        # 定义卫星图的变换
        sat_transforms_list = [
            transforms.Resize(self.sat_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), # 转换为 [0, 1] 的 FloatTensor
            #transforms.Lambda(lambda x: x * 255.0) # 转换为 [0, 255] 范围, 供模型使用
            transforms.Lambda(scale_to_255)
        ]
        
        # 定义全景图的变换
        pano_transforms_list = [
            transforms.Resize(self.pano_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor() # 转换为 [0, 1] 的 FloatTensor, 供损失函数使用
        ]
        
        # <--- [新增] 定义天空掩码的变换
        sky_transforms_list = [
            transforms.Resize(self.pano_size, interpolation=transforms.InterpolationMode.NEAREST), # 掩码使用最近邻插值
            transforms.Grayscale(num_output_channels=1), # 确保为单通道
            transforms.ToTensor() # 转换为 [0, 1] 的 FloatTensor
        ]

        self.sat_transform = transforms.Compose(sat_transforms_list)
        self.pano_transform = transforms.Compose(pano_transforms_list)
        self.sky_transform = transforms.Compose(sky_transforms_list) # <--- [新增]

    def _find_image_triplets(self): # <--- [修改] 函数名
        """
        自动扫描目录以查找匹配的 (卫星图, 全景图, 天空掩码) 对。
        通过匹配相同的文件基本名。
        """
        triplets = [] # <--- [修改]
        if not os.path.exists(self.sat_dir) or \
           not os.path.exists(self.pano_dir) or \
           not os.path.exists(self.sky_dir): # <--- [新增] 检查 sky_dir
            print(f"警告: 目录 {self.sat_dir}, {self.pano_dir} 或 {self.sky_dir} 不存在。")
            return []
            
        # 创建 (基本名 -> 完整文件名) 的映射
        sat_map = {os.path.splitext(f)[0]: f for f in os.listdir(self.sat_dir)}
        pano_map = {os.path.splitext(f)[0]: f for f in os.listdir(self.pano_dir)}
        sky_map = {os.path.splitext(f)[0]: f for f in os.listdir(self.sky_dir)} # <--- [新增]
        
        # 找出共同的基本名
        common_basenames = sorted(list(
            set(sat_map.keys()).intersection(set(pano_map.keys())).intersection(set(sky_map.keys())) # <--- [修改]
        ))
        
        for base_name in common_basenames:
            sat_file = sat_map[base_name]
            pano_file = pano_map[base_name]
            sky_file = sky_map[base_name] # <--- [新增]
            triplets.append((sat_file, pano_file, sky_file)) # <--- [修改]
            
        print(f"在 {self.data_split_dir} 中找到 {len(triplets)} 个图像三元组。")
        return triplets # <--- [修改]

    def __len__(self):
        return len(self.image_triplets) # <--- [修改]

    def __getitem__(self, idx):
        sat_name, pano_name, sky_name = self.image_triplets[idx] # <--- [修改]
        
        # 构建绝对路径
        sat_full_path = os.path.join(self.sat_dir, sat_name)
        pano_full_path = os.path.join(self.pano_dir, pano_name)
        sky_full_path = os.path.join(self.sky_dir, sky_name) # <--- [新增]
        
        try:
            # 加载图像
            sat_img = Image.open(sat_full_path).convert('RGB')
            pano_img = Image.open(pano_full_path).convert('RGB')
            sky_mask = Image.open(sky_full_path) # <--- [新增] (Grayscale transform 会处理通道)
            
        except FileNotFoundError as e:
            print(f"错误: 无法加载图像 {e}")
            return torch.zeros(3, *self.sat_size), \
                   torch.zeros(3, *self.pano_size), \
                   torch.zeros(1, *self.pano_size) # <--- [修改]

        # 应用变换
        sat_tensor = self.sat_transform(sat_img)
        pano_tensor = self.pano_transform(pano_img)
        sky_tensor = self.sky_transform(sky_mask) # <--- [新增]
        
        return sat_tensor, pano_tensor, sky_tensor # <--- [修改]

def get_loaders(opt: edict) -> (DataLoader, DataLoader):
    """
    根据配置创建训练和验证数据加载器。
    """
    
    # 训练集
    train_dataset = SatPanoDataset(
        opt=opt,
        data_split=opt.data.train_split,
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.train.batch_size,
        shuffle=True,
        num_workers=opt.train.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 验证集
    val_dataset = SatPanoDataset(
        opt=opt,
        data_split=opt.data.val_split,
        is_train=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.train.batch_size,
        shuffle=False,
        num_workers=opt.train.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"数据加载器创建完成。")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    return train_loader, val_loader

# -------------------------------------------------------------------
# (可选) 用于快速测试数据加载器的 main 块
# -------------------------------------------------------------------
def _create_dummy_data_for_loader_test():
    """创建一个临时的数据集结构用于测试。"""
    print("创建用于测试的虚拟数据...")
    root = 'dummy_dataset_loader_test'
    
    os.makedirs(os.path.join(root, 'train', 'sat_images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'train', 'pano_images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'train', 'sky_masks'), exist_ok=True) # <--- [新增]
    os.makedirs(os.path.join(root, 'val', 'sat_images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'val', 'pano_images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'val', 'sky_masks'), exist_ok=True) # <--- [新增]
    
    dummy_sat = Image.new('RGB', (512, 512), color='blue')
    dummy_pano = Image.new('RGB', (1024, 512), color='red')
    dummy_sky = Image.new('L', (1024, 512), color=255) # <--- [新增] L 模式, 全白 (天空)
    
    dummy_sat.save(os.path.join(root, 'train/sat_images/scene1.png'))
    dummy_pano.save(os.path.join(root, 'train/pano_images/scene1.png'))
    dummy_sky.save(os.path.join(root, 'train/sky_masks/scene1.png')) # <--- [新增]
    
    dummy_sat.save(os.path.join(root, 'train/sat_images/scene2.jpg'))
    dummy_pano.save(os.path.join(root, 'train/pano_images/scene2.jpg'))
    dummy_sky.save(os.path.join(root, 'train/sky_masks/scene2.jpg')) # <--- [新增]
    
    dummy_sat.save(os.path.join(root, 'val/sat_images/val_scene.png'))
    dummy_pano.save(os.path.join(root, 'val/pano_images/val_scene.jpg'))
    dummy_sky.save(os.path.join(root, 'val/sky_masks/val_scene.jpg')) # <--- [新增]
    
    print("虚拟数据创建完成。")
    return root

if __name__ == '__main__':
    # --- 1. 创建虚拟数据 ---
    dummy_root = _create_dummy_data_for_loader_test()

    # --- 2. 创建测试配置 ---
    opt_test = edict()
    opt_test.data = edict()
    opt_test.data.root_dir = dummy_root
    
    opt_test.data.train_split = 'train'
    opt_test.data.val_split = 'val'
    
    opt_test.data.sat_size = [256, 256] # H, W
    opt_test.data.pano_size = [512, 128] # W, H
    
    opt_test.train = edict()
    opt_test.train.batch_size = 1
    opt_test.train.num_workers = 0

    # --- 3. 测试加载器 ---
    print("正在测试 get_loaders...")
    train_loader, val_loader = get_loaders(opt_test)
    
    # --- 4. 提取一个批次 ---
    print("正在从 train_loader 提取一个批次...")
    sat_batch, pano_batch, sky_batch = next(iter(train_loader)) # <--- [修改]
    
    print(f"卫星图批次 shape: {sat_batch.shape}")
    print(f"全景图批次 shape: {pano_batch.shape}")
    print(f"天空掩码批次 shape: {sky_batch.shape}") # <--- [新增]
    
    print("\nDataLoader (带天空掩码) 测试成功!")

    # 清理 (可选)
    import shutil
    shutil.rmtree(dummy_root)
    print(f"已清理虚拟数据: {dummy_root}")