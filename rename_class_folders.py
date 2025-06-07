import os
import shutil
from pathlib import Path

def rename_class_folders(data_dir):
    """
    重命名类别文件夹：class_001-196 → class_000-195
    """
    data_path = Path(data_dir)
    
    for split in ['train', 'val']:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"警告: {split} 目录不存在")
            continue
            
        print(f"\n处理 {split} 目录:")
        
        # 获取所有类别文件夹
        class_folders = []
        for folder in split_dir.iterdir():
            if folder.is_dir() and folder.name.startswith('class_'):
                try:
                    class_id = int(folder.name.split('_')[1])
                    class_folders.append((class_id, folder))
                except ValueError:
                    continue
        
        # 按类别ID排序
        class_folders.sort()
        print(f"找到 {len(class_folders)} 个类别文件夹")
        
        # 先重命名为临时名称，避免冲突
        temp_folders = []
        for class_id, folder in class_folders:
            temp_name = f"temp_class_{class_id:03d}"
            temp_path = split_dir / temp_name
            folder.rename(temp_path)
            temp_folders.append((class_id, temp_path))
        
        print("临时重命名完成，开始最终重命名...")
        
        # 重命名为最终名称 (class_id - 1)
        for class_id, temp_folder in temp_folders:
            new_class_id = class_id - 1  # 1-196 → 0-195
            final_name = f"class_{new_class_id:03d}"
            final_path = split_dir / final_name
            temp_folder.rename(final_path)
            
            if class_id <= 5 or class_id > 191:  # 只显示前5个和后5个
                print(f"  {temp_folder.name} → {final_name}")
        
        if len(class_folders) > 10:
            print(f"  ... (共 {len(class_folders)} 个文件夹)")
    
    print(f"\n✓ 类别文件夹重命名完成!")
    print(f"✓ 类别范围: class_000 到 class_195")
    print(f"✓ 对应原始类别: 1 到 196")

def verify_class_folders(data_dir):
    """
    验证类别文件夹重命名结果
    """
    data_path = Path(data_dir)
    
    print("\n验证结果:")
    for split in ['train', 'val']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        class_folders = []
        for folder in split_dir.iterdir():
            if folder.is_dir() and folder.name.startswith('class_'):
                try:
                    class_id = int(folder.name.split('_')[1])
                    class_folders.append(class_id)
                except ValueError:
                    continue
        
        class_folders.sort()
        print(f"{split}: {len(class_folders)} 个类别")
        print(f"  范围: class_{min(class_folders):03d} 到 class_{max(class_folders):03d}")
        print(f"  期望: class_000 到 class_195")
        
        if min(class_folders) == 0 and max(class_folders) == 195 and len(class_folders) == 196:
            print(f"  ✓ {split} 目录验证通过")
        else:
            print(f"  ✗ {split} 目录验证失败")

if __name__ == "__main__":
    data_dir = "data"
    
    print("开始重命名类别文件夹...")
    print("将 class_001-196 重命名为 class_000-195")
    
    # 重命名文件夹
    rename_class_folders(data_dir)
    
    # 验证结果
    verify_class_folders(data_dir)
    
    print("\n重要提醒:")
    print("✓ 类别标号已从1-196转换为0-195")
    print("✓ 训练时使用 --num-classes 196")
    print("✓ 文件夹名称: class_000 到 class_195")
    print("✓ PyTorch DataLoader会自动根据文件夹名称分配标签0-195") 