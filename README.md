# TNT Car Classification Project

🚗 A deep learning project for car classification using **TNT (Transformer in Transformer)** architecture on the Cars-196 dataset.

## 🎯 Project Overview

This project implements a TNT (Transformer in Transformer) model for fine-grained car classification. The model achieves high accuracy on the Cars-196 dataset by leveraging the power of transformer architecture with hierarchical visual representations.

## ✨ Features

- **TNT Architecture**: Implementation of Transformer in Transformer for vision tasks
- **Cars-196 Dataset**: Fine-grained classification of 196 car models
- **Complete Training Pipeline**: From data preprocessing to model evaluation
- **Inference Script**: Ready-to-use inference on new images
- **Pretrained Models**: Support for loading pretrained weights
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Advanced Augmentation**: Mixup, CutMix, and other data augmentation techniques

## 📊 Dataset

**Cars-196 Dataset**: Stanford Cars dataset containing 196 classes of cars
- **Training**: 8,144 images
- **Testing**: 8,041 images
- **Classes**: 196 different car models
- **Task**: Fine-grained classification

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/CAN-Lee/TNT-car.git
cd TNT-car

# Install dependencies
pip install torch torchvision
pip install timm
pip install pandas numpy pillow
```

## 📁 Project Structure

```
TNT-car/
├── train.py              # Main training script
├── tnt.py                # TNT model implementation
├── test_inference.py     # Inference script for testing
├── data/
│   ├── myloader.py       # Custom data loader
│   ├── organize_*.py     # Data organization scripts
│   ├── test.csv          # Test annotations
│   └── train/            # Training images (not included)
├── output/               # Training outputs and checkpoints
├── models/               # Pretrained model weights
└── .vscode/              # VSCode debug configurations
```

## 🚀 Usage

### Training

```bash
python train.py data \
    --model tnt_b_patch16_224 \
    --pretrain_path models/tnt_b_82.9.pth.tar \
    --num-classes 196 \
    --batch-size 16 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --epochs 50 \
    --warmup-epochs 5 \
    --sched cosine \
    --min-lr 1e-6 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --reprob 0.25 \
    --workers 4 \
    --pin-mem \
    --output ./output \
    --log-interval 10
```

### Evaluation Only

```bash
python train.py data \
    --model tnt_b_patch16_224 \
    --pretrain_path models/tnt_b_82.9.pth.tar \
    --num-classes 196 \
    --batch-size 16 \
    --workers 4 \
    --pin-mem \
    --evaluate
```

### Inference on Test Set

```bash
python test_inference.py \
    --model-path output/train/[timestamp]/model_best.pth.tar \
    --test-dir data/test \
    --csv-path data/test.csv \
    --model tnt_b_patch16_224 \
    --num-classes 196 \
    --batch-size 16
```

## 🏗️ Model Architecture

**TNT (Transformer in Transformer)**:
- **Outer Transformer**: Processes image patches
- **Inner Transformer**: Processes pixel-level features within each patch
- **Hierarchical Design**: Captures both local and global dependencies
- **Patch Size**: 16x16 pixels
- **Input Resolution**: 224x224 pixels

## 📈 Training Details

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 0.0001 with cosine annealing
- **Batch Size**: 16
- **Epochs**: 50
- **Warmup**: 5 epochs
- **Augmentations**: Mixup, CutMix, Random Erasing
- **Mixed Precision**: Enabled for faster training

## 🎯 Performance

The model achieves competitive results on the Cars-196 dataset:
- **Top-1 Accuracy**: Check training logs for specific numbers
- **Top-5 Accuracy**: Check training logs for specific numbers

## 🔧 Key Scripts

- **`train.py`**: Main training and evaluation script
- **`tnt.py`**: TNT model implementation
- **`test_inference.py`**: Inference script for new images
- **`organize_*.py`**: Data preprocessing utilities
- **`rename_class_folders.py`**: Class folder organization

## 📝 Data Preparation

1. **Download Cars-196 dataset**
2. **Organize data structure**:
   ```
   data/
   ├── train/
   │   ├── class_000/
   │   ├── class_001/
   │   └── ...
   ├── val/
   │   ├── class_000/
   │   ├── class_001/
   │   └── ...
   └── test/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

## 🐛 Debugging

VSCode debug configurations are provided in `.vscode/launch.json`:
- **"Debug Train.py"**: Full training mode
- **"Debug Evaluate Only"**: Evaluation mode only

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

⭐ If you find this project useful, please give it a star on GitHub! 