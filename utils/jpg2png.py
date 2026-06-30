import os
import argparse
from PIL import Image

def batch_convert_jpg_to_png(folder_path, delete_original=False):
    """
    在指定文件夹中将 JPG 转换为 PNG。
    :param folder_path: 目标文件夹路径
    :param delete_original: 是否删除原 JPG 文件 (默认 False，即保留原图)
    """
    
    # 确保路径存在
    if not os.path.exists(folder_path):
        print(f"错误：找不到文件夹路径 -> {folder_path}")
        return

    files = os.listdir(folder_path)
    count = 0
    
    print(f"正在处理文件夹: {folder_path}")
    print("-" * 30)

    for filename in files:
        # 检查是否为 jpg/jpeg (忽略大小写)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            
            # 获取文件名（不含后缀）
            name_without_ext = os.path.splitext(filename)[0]
            # 组合新的 png 路径
            new_filename = f"{name_without_ext}.png"
            new_path = os.path.join(folder_path, new_filename)

            # 如果已经存在同名 png，跳过以防覆盖
            if os.path.exists(new_path):
                print(f"[跳过] {new_filename} 已存在")
                continue

            try:
                with Image.open(full_path) as img:
                    img.save(new_path, 'PNG')
                    print(f"[转换] {filename} -> {new_filename}")
                    count += 1
                
                # 如果设置为删除原图，则执行删除
                if delete_original:
                    os.remove(full_path)
                    print(f"[删除] 原文件 {filename} 已删除")

            except Exception as e:
                print(f"[错误] 无法处理 {filename}: {e}")

    print("-" * 30)
    print(f"处理完成！成功转换 {count} 张图片。")

# ---在此处修改配置---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JPG/JPEG files in a folder to PNG.")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--delete_original", action="store_true")
    args = parser.parse_args()
    batch_convert_jpg_to_png(args.folder, args.delete_original)
