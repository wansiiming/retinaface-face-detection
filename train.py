import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import logging
from datetime import datetime

from train_config import TRAIN_CONFIG
from dataset import FaceDataset, collate_fn
from retinaface_model import RetinaFace
from losses import RetinaFaceLoss

def setup_logging(output_dir):
    log_dir = output_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = log_dir / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_kpt_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (images, bboxes, keypoints) in enumerate(train_loader):
        images = images.to(device)
        bboxes = bboxes.to(device)
        keypoints = keypoints.to(device)
        
        optimizer.zero_grad()
        
        cls_pred, bbox_pred, kpt_pred = model(images)
        
        loss, loss_info = criterion(cls_pred, bbox_pred, kpt_pred, bboxes, keypoints)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss_info['total_loss']
        total_cls_loss += loss_info['cls_loss']
        total_bbox_loss += loss_info['bbox_loss']
        total_kpt_loss += loss_info['kpt_loss']
        num_batches += 1
        
        if (batch_idx + 1) % TRAIN_CONFIG['log_interval'] == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / elapsed
            
            logger.info(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] "
                       f"Loss: {avg_loss:.4f} "
                       f"Cls: {total_cls_loss/num_batches:.4f} "
                       f"BBox: {total_bbox_loss/num_batches:.4f} "
                       f"Kpt: {total_kpt_loss/num_batches:.4f} "
                       f"Speed: {speed:.2f} img/s")
    
    avg_loss = total_loss / num_batches
    return {
        'loss': avg_loss,
        'cls_loss': total_cls_loss / num_batches,
        'bbox_loss': total_bbox_loss / num_batches,
        'kpt_loss': total_kpt_loss / num_batches
    }

def validate(model, val_loader, criterion, device, logger):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, bboxes, keypoints in val_loader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            keypoints = keypoints.to(device)
            
            cls_pred, bbox_pred, kpt_pred = model(images)
            loss, loss_info = criterion(cls_pred, bbox_pred, kpt_pred, bboxes, keypoints)
            
            total_loss += loss_info['total_loss']
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    output_dir = Path(TRAIN_CONFIG['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("RetinaFace Training Started")
    logger.info("="*60)
    
    set_seed(TRAIN_CONFIG['seed'])
    
    device = torch.device(TRAIN_CONFIG['device'])
    logger.info(f"Using device: {device}")
    
    logger.info("Loading datasets...")
    train_dataset = FaceDataset(
        TRAIN_CONFIG['data_dir'], 
        split='train', 
        target_size=TRAIN_CONFIG['input_size']
    )
    
    val_dataset = FaceDataset(
        TRAIN_CONFIG['data_dir'], 
        split='val', 
        target_size=TRAIN_CONFIG['input_size']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True, 
        num_workers=TRAIN_CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False, 
        num_workers=TRAIN_CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    logger.info("Building model...")
    model = RetinaFace(
        num_classes=TRAIN_CONFIG['model']['num_classes'],
        anchors_num=TRAIN_CONFIG['model']['anchors_num']
    )
    model = model.to(device)
    
    criterion = RetinaFaceLoss(
        cls_loss_weight=TRAIN_CONFIG['loss']['cls_loss_weight'],
        bbox_loss_weight=TRAIN_CONFIG['loss']['bbox_loss_weight'],
        kpt_loss_weight=TRAIN_CONFIG['loss']['kpt_loss_weight']
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    if TRAIN_CONFIG['lr_scheduler']['type'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=TRAIN_CONFIG['lr_scheduler']['step_size'],
            gamma=TRAIN_CONFIG['lr_scheduler']['gamma']
        )
    elif TRAIN_CONFIG['lr_scheduler']['type'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_CONFIG['num_epochs']
        )
    else:
        scheduler = None
    
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{TRAIN_CONFIG['num_epochs']}")
        logger.info("-" * 60)
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        logger.info(f"Training - Loss: {train_metrics['loss']:.4f}, "
                   f"Cls: {train_metrics['cls_loss']:.4f}, "
                   f"BBox: {train_metrics['bbox_loss']:.4f}, "
                   f"Kpt: {train_metrics['kpt_loss']:.4f}")
        
        if len(val_loader) > 0:
            val_loss = validate(model, val_loader, criterion, device, logger)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
        
        if epoch % TRAIN_CONFIG['save_interval'] == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("="*60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
