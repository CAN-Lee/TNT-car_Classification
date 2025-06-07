import os
import pandas as pd
import shutil
from pathlib import Path
from collections import defaultdict
import random

def organize_cars_by_class_corrected(csv_file, source_dir, target_dir, train_ratio=0.8):
    """
    根据CSV文件中的类别标签组织汽车数据，并将类别从1-196转换为0-195
    
    Args:
        csv_file: train.csv文件路径
        source_dir: 原始图片目录 (data/cars)
        target_dir: 目标数据目录 (data)
        train_ratio: 训练集比例 (默认0.8)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"CSV文件包含 {len(df)} 行数据")
    
    # 将类别从1-196转换为0-195
    df['Class_Corrected'] = df['Class'] - 1
    print(f"类别转换: {df['Class'].min()}-{df['Class'].max()} → {df['Class_Corrected'].min()}-{df['Class_Corrected'].max()}")
    
    # 获取唯一的类别（转换后）
    unique_classes = sorted(df['Class_Corrected'].unique())
    print(f"发现 {len(unique_classes)} 个类别: 0-{max(unique_classes)}")
    
    # 按类别分组
    class_groups = df.groupby('Class_Corrected')
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 统计信息
    total_train = 0
    total_val = 0
    
    # 设置随机种子
    random.seed(42)
    
    for class_id in unique_classes:
        class_data = class_groups.get_group(class_id)
        class_images = class_data['image'].tolist()
        original_class_id = class_data['Class'].iloc[0]  # 原始类别ID
        
        print(f"\n处理类别 {original_class_id}→{class_id}: {len(class_images)} 张图片")
        
        # 创建类别目录（使用转换后的类别ID）
        train_class_dir = target_path / "train" / f"class_{class_id:03d}"
        val_class_dir = target_path / "val" / f"class_{class_id:03d}"
        
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # 随机打乱该类别的图片
        random.shuffle(class_images)
        
        # 计算分割点
        split_idx = int(len(class_images) * train_ratio)
        train_images = class_images[:split_idx]
        val_images = class_images[split_idx:]
        
        # 复制训练集图片
        for img_name in train_images:
            src_path = source_path / img_name
            if src_path.exists():
                dst_path = train_class_dir / img_name
                shutil.copy2(src_path, dst_path)
                total_train += 1
            else:
                print(f"警告: 图片文件不存在: {img_name}")
        
        # 复制验证集图片
        for img_name in val_images:
            src_path = source_path / img_name
            if src_path.exists():
                dst_path = val_class_dir / img_name
                shutil.copy2(src_path, dst_path)
                total_val += 1
            else:
                print(f"警告: 图片文件不存在: {img_name}")
        
        print(f"  训练集: {len(train_images)} 张")
        print(f"  验证集: {len(val_images)} 张")
    
    print(f"\n数据整理完成!")
    print(f"总计:")
    print(f"  训练集: {total_train} 张图片")
    print(f"  验证集: {total_val} 张图片")
    print(f"  类别数量: {len(unique_classes)} (0-{max(unique_classes)})")
    
    # 显示目录结构
    print(f"\n创建的目录结构:")
    print(f"├── train/")
    for class_id in unique_classes[:5]:  # 只显示前5个
        class_data = class_groups.get_group(class_id)
        train_count = int(len(class_data) * train_ratio)
        print(f"│   ├── class_{class_id:03d}/ ({train_count} 张图片)")
    if len(unique_classes) > 5:
        print(f"│   ├── ...")
        print(f"│   └── class_{max(unique_classes):03d}/")
    print(f"└── val/")
    for class_id in unique_classes[:5]:  # 只显示前5个
        class_data = class_groups.get_group(class_id)
        val_count = len(class_data) - int(len(class_data) * train_ratio)
        print(f"    ├── class_{class_id:03d}/ ({val_count} 张图片)")
    if len(unique_classes) > 5:
        print(f"    ├── ...")
        print(f"    └── class_{max(unique_classes):03d}/")

def create_corrected_class_mapping(csv_file, output_file="data/class_mapping_corrected.txt"):
    """
    创建修正后的类别映射文件
    """
    df = pd.read_csv(csv_file)
    unique_classes = sorted(df['Class'].unique())
    
    with open(output_file, 'w') as f:
        f.write("原始类别ID -> 修正类别ID -> 文件夹名称\n")
        f.write("=" * 50 + "\n")
        for original_class_id in unique_classes:
            corrected_class_id = original_class_id - 1
            f.write(f"{original_class_id:3d} -> {corrected_class_id:3d} -> class_{corrected_class_id:03d}\n")
    
    print(f"修正后的类别映射文件已保存到: {output_file}")

if __name__ == "__main__":
    # 设置路径
    current_dir = os.getcwd()
    csv_file = os.path.join(current_dir, "data", "train.csv")
    source_dir = os.path.join(current_dir, "data", "cars")
    target_dir = os.path.join(current_dir, "data")
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        exit(1)
    
    if not os.path.exists(source_dir):
        print(f"错误: 图片目录不存在: {source_dir}")
        exit(1)
    
    # 执行数据整理
    organize_cars_by_class_corrected(csv_file, source_dir, target_dir, train_ratio=0.8)
    
    # 创建修正后的类别映射文件
    create_corrected_class_mapping(csv_file, "data/class_mapping_corrected.txt")
    
    print("\n重要提醒:")
    print("✓ 类别标号已从1-196转换为0-195")
    print("✓ 训练时使用 --num-classes 196")
    print("✓ 文件夹名称: class_000 到 class_195") 