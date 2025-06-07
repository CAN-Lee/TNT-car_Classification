import os
import pandas as pd
import shutil
from pathlib import Path

def organize_test_data(test_csv_file, source_dir, target_dir):
    """
    根据test.csv文件组织测试数据
    
    Args:
        test_csv_file: test.csv文件路径
        source_dir: 原始图片目录 (data/cars)
        target_dir: 目标数据目录 (data)
    """
    # 读取CSV文件
    df = pd.read_csv(test_csv_file)
    print(f"test.csv文件包含 {len(df)} 行数据")
    
    # 获取所有测试图片文件名
    test_images = df['image'].tolist()
    print(f"发现 {len(test_images)} 个测试图片")
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建test目录
    test_dir = target_path / "test" / "test_images"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    copied_count = 0
    missing_count = 0
    missing_files = []
    
    print("正在复制测试集文件...")
    
    # 复制测试集图片
    for img_name in test_images:
        src_path = source_path / img_name
        if src_path.exists():
            dst_path = test_dir / img_name
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_files.append(img_name)
            missing_count += 1
            print(f"警告: 图片文件不存在: {img_name}")
    
    print(f"\n测试数据整理完成!")
    print(f"成功复制: {copied_count} 个文件")
    if missing_count > 0:
        print(f"缺失文件: {missing_count} 个")
        
    print(f"\n创建的目录结构:")
    print(f"test/")
    print(f"└── test_images/ ({copied_count} 个文件)")
    
    # 如果有缺失文件，保存到日志
    if missing_files:
        log_file = target_path / "missing_test_files.txt"
        with open(log_file, 'w') as f:
            f.write("缺失的测试文件:\n")
            f.write("=" * 30 + "\n")
            for missing_file in missing_files:
                f.write(f"{missing_file}\n")
        print(f"缺失文件列表已保存到: {log_file}")
    
    return copied_count, missing_count

def create_test_list(test_csv_file, output_file="data/test_list.txt"):
    """
    创建测试图片列表文件
    """
    df = pd.read_csv(test_csv_file)
    test_images = df['image'].tolist()
    
    with open(output_file, 'w') as f:
        f.write("测试图片列表 (按CSV文件顺序)\n")
        f.write("=" * 40 + "\n")
        for i, img_name in enumerate(test_images, 1):
            f.write(f"{i:4d}. {img_name}\n")
    
    print(f"测试图片列表已保存到: {output_file}")

if __name__ == "__main__":
    # 设置路径
    current_dir = os.getcwd()
    test_csv_file = os.path.join(current_dir, "data", "test.csv")
    source_dir = os.path.join(current_dir, "data", "cars")
    target_dir = os.path.join(current_dir, "data")
    
    # 检查文件是否存在
    if not os.path.exists(test_csv_file):
        print(f"错误: test.csv文件不存在: {test_csv_file}")
        exit(1)
    
    if not os.path.exists(source_dir):
        print(f"错误: 图片目录不存在: {source_dir}")
        exit(1)
    
    # 执行数据整理
    copied_count, missing_count = organize_test_data(test_csv_file, source_dir, target_dir)
    
    # 创建测试图片列表
    create_test_list(test_csv_file, "data/test_list.txt")
    
    print(f"\n总结:")
    print(f"- 成功处理测试图片: {copied_count} 个")
    print(f"- 缺失图片: {missing_count} 个")
    print(f"- 测试数据可用于模型推理") 