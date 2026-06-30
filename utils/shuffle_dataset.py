import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def shuffle_copy_images(src_dir, dst_dir):
    """
    将 src_dir 中的图片复制到 dst_dir，并随机打乱文件名。
    """
    src = Path(src_dir)
    dst = Path(dst_dir)

    # 1. 基础检查
    if not src.exists():
        print(f"❌ 错误：源文件夹不存在 -> {src}")
        return
    
    if src == dst:
        print("❌ 错误：源文件夹和目标文件夹不能相同！为了保护原数据，请指定一个新的文件夹位置。")
        return

    # 创建目标文件夹 (如果不存在)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"📂 目标文件夹已准备: {dst}")

    # 2. 获取源文件并按扩展名分组
    #    逻辑：只在相同格式之间互换文件名 (jpg换jpg, png换png)
    files = [f for f in src.iterdir() if f.is_file()]
    ext_groups = {}
    
    for f in files:
        ext = f.suffix.lower()
        # 过滤非图片文件 (可选，这里列出常见图片格式)
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            continue
            
        if ext not in ext_groups:
            ext_groups[ext] = []
        ext_groups[ext].append(f)

    if not ext_groups:
        print("⚠️  源文件夹中没有找到图片文件。")
        return

    print(f"🔍 扫描完成，即将处理...")

    total_count = 0
    
    # 3. 执行打乱并复制
    for ext, src_file_list in ext_groups.items():
        count = len(src_file_list)
        if count == 0: continue
        
        print(f"🔄 正在处理 {ext} 格式 ({count} 张)...")

        # 提取文件名列表
        original_names = [f.name for f in src_file_list]
        
        # 打乱文件名列表
        # logic: 原始文件列表不变，目标名字列表打乱
        # result: file_1.jpg 的内容会被复制并保存为 file_random.jpg
        shuffled_names = original_names.copy()
        random.shuffle(shuffled_names)

        # 使用 tqdm 显示进度条
        for src_file, target_name in tqdm(zip(src_file_list, shuffled_names), total=count):
            target_path = dst / target_name
            
            # 执行复制操作 (copy2 保留文件元数据如时间戳)
            shutil.copy2(src_file, target_path)
            total_count += 1

    print("\n" + "="*40)
    print(f"✅ 完成！")
    print(f"📂 源文件夹 (未修改): {src}")
    print(f"📂 新文件夹 (已打乱): {dst}")
    print(f"📄 共处理图片: {total_count} 张")
    print("="*40)

if __name__ == "__main__":
    # 交互式输入
    print("--- 图片文件名随机打乱工具 (复制模式) ---")
    src_input = input("请输入【源】图片文件夹路径: ").strip().strip('"').strip("'")
    dst_input = input("请输入【保存结果】的文件夹路径: ").strip().strip('"').strip("'")
    
    shuffle_copy_images(src_input, dst_input)