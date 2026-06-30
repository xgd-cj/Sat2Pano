import os
import argparse
from PIL import Image
from tqdm import tqdm

def convert_all_to_png(folder_path):
    """
    将文件夹内的所有图片转换为 PNG 格式，并删除原文件。
    """
    if not os.path.exists(folder_path):
        print(f"错误：文件夹路径不存在 -> {folder_path}")
        return

    # 获取所有文件列表
    files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
    
    print(f"正在扫描文件夹: {folder_path}")
    print(f"文件总数: {len(files)}")

    # 支持转换的源格式
    valid_extensions = {'.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    
    converted_count = 0
    skipped_count = 0
    error_count = 0

    for filename in tqdm(files, desc="Converting"):
        file_root, file_ext = os.path.splitext(filename)
        original_path = os.path.join(folder_path, filename)
        target_path = os.path.join(folder_path, file_root + ".png")

        # 1. 如果已经是 png，跳过
        if file_ext.lower() == '.png':
            skipped_count += 1
            continue

        # 2. 如果是其他图片格式，进行转换
        if file_ext.lower() in valid_extensions:
            try:
                with Image.open(original_path) as img:
                    # 转换为 RGB (防止 CMYK 或 RGBA 导致问题，视需求而定，通常RGB最稳)
                    img = img.convert('RGB')
                    img.save(target_path, 'PNG')
                
                # 3. 转换成功后，删除原文件 (重要！否则文件夹里会有两份文件)
                os.remove(original_path)
                converted_count += 1
                
            except Exception as e:
                print(f"\n[Error] 转换失败: {filename} -> {e}")
                error_count += 1
        else:
            # 非图片文件，忽略
            pass

    print("\n" + "="*30)
    print("处理完成！")
    print(f"✅ 成功转换: {converted_count}")
    print(f"⏩ 跳过(已是PNG): {skipped_count}")
    print(f"❌ 失败: {error_count}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images in a folder to PNG.")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--yes", action="store_true", help="Run without interactive confirmation.")
    args = parser.parse_args()
    target_dir = args.folder
    
    # 二次确认，防止误删
    print(f"警告：该操作将把 {target_dir} 下的所有 .jpg/.jpeg 图片转换为 .png 并【删除原文件】。")
    confirm = "y" if args.yes else input("确认继续吗？(输入 y 继续): ")
    
    if confirm.lower() == 'y':
        convert_all_to_png(target_dir)
    else:
        print("操作已取消。")
