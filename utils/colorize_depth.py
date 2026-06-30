import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

def apply_attention_colormap(image_path, output_dir, colormap='magma'):
    """
    处理单张图片并保存。
    """
    # 1. 读取图片
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"[Error] 无法读取图片: {image_path}")
        return

    # 2. 归一化 (0-255 -> 0.0-1.0)
    norm_img = img_gray / 255.0

    # 3. 应用色谱
    try:
        cmap = plt.get_cmap(colormap)
    except ValueError:
        cmap = plt.get_cmap('magma')

    colored_img = cmap(norm_img)
    colored_img = np.uint8(colored_img * 255)
    colored_img_bgr = cv2.cvtColor(colored_img, cv2.COLOR_RGBA2BGR)

    # 4. 路径处理
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, file_name)

    # 5. 保存
    cv2.imwrite(save_path, colored_img_bgr)

def batch_process_depth_images(input_folder, output_folder, colormap='magma'):
    """
    批量扫描文件夹中所有 _depth.png 后缀的文件并处理。
    """
    # 构建搜索模式，匹配所有以 _depth.png 结尾的文件
    search_pattern = os.path.join(input_folder, "*_depth.png")
    image_list = glob.glob(search_pattern)
    
    if not image_list:
        print(f"[Warning] 在 {input_folder} 中未找到任何以 _depth.png 结尾的图片。")
        return

    print(f"[Info] 找到 {len(image_list)} 张深度图，开始处理...")

    for img_path in image_list:
        apply_attention_colormap(img_path, output_folder, colormap)
        print(f"  >> 已处理: {os.path.basename(img_path)}")

    print(f"[Success] 全部处理完成！结果保存至: {output_folder}")

def parse_args():
    parser = argparse.ArgumentParser(description="Apply a matplotlib colormap to *_depth.png files.")
    parser.add_argument("--input_dir", type=str, default="outputs/video/S1")
    parser.add_argument("--output_dir", type=str, default="outputs/video/S1_depth_color")
    parser.add_argument("--colormap", type=str, default="magma")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_process_depth_images(args.input_dir, args.output_dir, colormap=args.colormap)
