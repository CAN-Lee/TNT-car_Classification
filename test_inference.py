#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from timm.models import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse

# Import custom TNT model
import tnt

def create_test_dataloader(test_dir, data_config, batch_size=16):
    """Create a dataloader for test images"""
    from torch.utils.data import Dataset, DataLoader
    
    class TestDataset(Dataset):
        def __init__(self, test_dir, transform):
            self.test_dir = test_dir
            self.transform = transform
            self.image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
            self.image_files.sort()  # 确保顺序一致
            
        def __len__(self):
            return len(self.image_files)
            
        def __getitem__(self, idx):
            img_path = os.path.join(self.test_dir, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, self.image_files[idx]
    
    # 创建transform
    transform = create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config['crop_pct']
    )
    
    dataset = TestDataset(test_dir, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 不打乱，保持文件名顺序
        num_workers=4
    )
    
    return dataloader

def main():
    parser = argparse.ArgumentParser(description='Test inference on individual images')
    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to the trained model')
    parser.add_argument('--test-dir', default='data/test', type=str,
                        help='Path to test images directory')
    parser.add_argument('--csv-path', default='data/test.csv', type=str,
                        help='Path to test.csv file to update')
    parser.add_argument('--model', default='tnt_b_patch16_224', type=str,
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=196,
                        help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    # 创建模型 - 使用自定义TNT模型
    print(f"Creating model: {args.model}")
    
    try:
        # 首先尝试timm库的模型
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            checkpoint_path=None
        )
    except RuntimeError:
        # 如果timm库没有这个模型，使用自定义TNT模型
        print(f"Model {args.model} not found in timm, using custom TNT model...")
        if args.model == 'tnt_b_patch16_224':
            model = tnt.tnt_b_patch16_224(num_classes=args.num_classes)
        else:
            raise ValueError(f"Unknown custom model: {args.model}")
    
    # 加载权重
    print(f"Loading weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 提取模型权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 移除模型前缀（如果存在）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    print("Model weights loaded successfully!")
    
    # 移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 获取数据配置 - 使用默认配置如果模型没有config
    try:
        data_config = resolve_data_config({}, model=model)
    except:
        # 使用TNT模型的默认配置
        data_config = {
            'input_size': (3, 224, 224),
            'interpolation': 'bicubic',
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'crop_pct': 0.9
        }
    
    print(f"Data config: {data_config}")
    
    # 创建测试数据加载器
    print(f"Loading test data from: {args.test_dir}")
    test_loader = create_test_dataloader(args.test_dir, data_config, args.batch_size)
    
    # 进行推理
    predictions = []
    filenames = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, (images, image_names) in enumerate(test_loader):
            images = images.to(device)
            
            outputs = model(images)
            
            # 获取预测类别
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            filenames.extend(image_names)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    print(f"Inference completed! Processed {len(predictions)} images.")
    
    # 读取现有的CSV文件
    print(f"Reading CSV file: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # 创建文件名到预测的映射
    filename_to_pred = dict(zip(filenames, predictions))
    
    # 更新Class列（预测结果+1，因为模型输出是0-195，但实际类别是1-196）
    updated_count = 0
    for idx, row in df.iterrows():
        image_name = row['image']
        if image_name in filename_to_pred:
            df.at[idx, 'Class'] = filename_to_pred[image_name] + 1  # +1转换为1-196
            updated_count += 1
    
    print(f"Updated {updated_count} rows out of {len(df)} total rows.")
    
    # 保存更新后的CSV文件
    output_csv = args.csv_path.replace('.csv', '_with_predictions.csv')
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
    
    # 显示一些统计信息
    print(f"\nPrediction statistics:")
    if len(predictions) > 0:
        print(f"Class distribution: {np.bincount(np.array(predictions) + 1)}")  # +1显示实际类别分布
        print(f"Min class: {min(predictions) + 1}, Max class: {max(predictions) + 1}")

if __name__ == '__main__':
    main()