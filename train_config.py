import torch

TRAIN_CONFIG = {
    'data_dir': r'C:\Users\王子鸣\OneDrive\桌面\FaceDataset_organized',
    'output_dir': r'C:\Users\王子鸣\OneDrive\桌面\深度学习实训\checkpoints',
    
    'num_epochs': 100,
    'batch_size': 8,
    'num_workers': 0,
    
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    
    'lr_scheduler': {
        'type': 'StepLR',
        'step_size': 20,
        'gamma': 0.1
    },
    
    'input_size': 640,
    
    'model': {
        'backbone': 'resnet50',
        'pretrained': True,
        'num_classes': 2,
        'anchors_num': 2
    },
    
    'loss': {
        'cls_loss_weight': 1.0,
        'bbox_loss_weight': 1.0,
        'kpt_loss_weight': 1.0
    },
    
    'save_interval': 5,
    'log_interval': 10,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'seed': 42
}

if __name__ == '__main__':
    print("=" * 60)
    print("RetinaFace Training Configuration")
    print("=" * 60)
    for key, value in TRAIN_CONFIG.items():
        print(f"{key}: {value}")
    print("=" * 60)
