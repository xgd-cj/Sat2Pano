import os
import shutil
import argparse

# ==================== 【配置区域】 (请在此处修改) ====================

# 1. 源文件夹路径 (可以使用绝对路径，例如 r"D:\Data\CVUSA_results-full")
# r"" 表示原始字符串，防止 Windows 路径中的斜杠被转义
SOURCE_DIR = r"outputs/stage1_inference"

# 2. 自定义输出配置
# 格式: { "要识别的后缀": "保存的目标文件夹路径" }
# 你可以将路径修改为电脑上的任何位置，例如 r"E:\MyDataset\RGB_Images"
TARGET_CONFIG = {
    "_depth.png":   r"data/stage2train/depth",   
    "_rgb.png":     r"data/stage2train/init_proj",
    "_opacity.png": r"data/stage2train/opacity"
}

# 3. 自定义文件名生成函数
# 输入: 原始文件名 (例如 "0000003_rgb.png"), 匹配到的后缀 (例如 "_rgb.png")
# 输出: 你想要保存的新文件名
def generate_new_filename(original_name, suffix):
    # ---------------------------------------------------------
    # 策略 A: 去除后缀 (当前需求)
    # 逻辑: 将 "_rgb.png" 替换为 ".png"
    # 结果: "0000003_rgb.png" -> "0000003.png"
    return original_name.replace(suffix, ".png")
    
    # 策略 B (备用): 如果你想保留原始名字，只需返回 original_name
    # return original_name
    # ---------------------------------------------------------

# ==================== 【代码执行区域】 (通常无需修改) ====================

def main(source_dir=SOURCE_DIR, target_config=None):
    if target_config is None:
        target_config = TARGET_CONFIG
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 找不到源文件夹 -> {source_dir}")
        return

    count = 0
    print(f"正在扫描目录: {source_dir} ...")

    # 遍历源目录中的文件
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        
        # 确保是文件而不是文件夹
        if not os.path.isfile(src_path):
            continue

        # 遍历配置，检查文件是否符合某一类后缀
        for suffix, target_folder in target_config.items():
            if filename.endswith(suffix):
                # 1. 确保目标文件夹存在
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    print(f"创建新文件夹: {target_folder}")

                # 2. 生成新文件名 (调用上面的配置函数)
                new_filename = generate_new_filename(filename, suffix)
                dst_path = os.path.join(target_folder, new_filename)

                # 3. 复制文件
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"[复制成功] {filename} -> {target_folder}/{new_filename}")
                    count += 1
                except Exception as e:
                    print(f"[复制失败] {filename}: {e}")
                
                # 找到匹配后跳出内层循环（假设一个文件只属于一类）
                break

    print("=" * 30)
    print(f"处理完成！共处理了 {count} 个文件。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split Stage 1 outputs into Stage 2 init/depth/opacity folders.")
    parser.add_argument("--source_dir", type=str, default=SOURCE_DIR)
    parser.add_argument("--depth_dir", type=str, default=TARGET_CONFIG["_depth.png"])
    parser.add_argument("--init_proj_dir", type=str, default=TARGET_CONFIG["_rgb.png"])
    parser.add_argument("--opacity_dir", type=str, default=TARGET_CONFIG["_opacity.png"])
    args = parser.parse_args()
    config = {
        "_depth.png": args.depth_dir,
        "_rgb.png": args.init_proj_dir,
        "_opacity.png": args.opacity_dir,
    }
    main(args.source_dir, config)
