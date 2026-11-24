import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinaFaceLoss(nn.Module):
    def __init__(self, cls_loss_weight=1.0, bbox_loss_weight=1.0, kpt_loss_weight=1.0):
        super(RetinaFaceLoss, self).__init__()
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.kpt_loss_weight = kpt_loss_weight
        
        self.cls_criterion = FocalLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        
    def forward(self, cls_pred, bbox_pred, kpt_pred, bbox_target, kpt_target):
        batch_size = cls_pred.size(0)
        
        cls_target = torch.ones(batch_size, cls_pred.size(1), dtype=torch.long, device=cls_pred.device)
        
        cls_loss = self.cls_criterion(cls_pred, cls_target)
        cls_loss = cls_loss.mean()
        
        valid_bbox_mask = (bbox_target.sum(dim=1) > 0).float()
        if valid_bbox_mask.sum() > 0:
            bbox_loss = self.smooth_l1(bbox_pred * valid_bbox_mask.unsqueeze(-1), 
                                       bbox_target * valid_bbox_mask.unsqueeze(-1))
            bbox_loss = bbox_loss * valid_bbox_mask.mean()
        else:
            bbox_loss = torch.tensor(0.0, device=cls_pred.device)
        
        valid_kpt_mask = (kpt_target.sum(dim=1).sum(dim=1) > 0).float()
        if valid_kpt_mask.sum() > 0:
            kpt_loss = self.smooth_l1(kpt_pred * valid_kpt_mask.unsqueeze(-1), 
                                      kpt_target * valid_kpt_mask.unsqueeze(-1))
            kpt_loss = kpt_loss * valid_kpt_mask.mean()
        else:
            kpt_loss = torch.tensor(0.0, device=cls_pred.device)
        
        total_loss = (self.cls_loss_weight * cls_loss + 
                     self.bbox_loss_weight * bbox_loss + 
                     self.kpt_loss_weight * kpt_loss)
        
        loss_info = {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'bbox_loss': bbox_loss.item(),
            'kpt_loss': kpt_loss.item()
        }
        
        return total_loss, loss_info

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        
    def forward(self, pred, target):
        x = torch.abs(pred - target)
        wing_loss = torch.where(
            x < self.omega,
            self.omega * torch.log(1 + x / self.epsilon),
            x - self.C
        )
        return wing_loss.mean()

if __name__ == '__main__':
    criterion = RetinaFaceLoss()
    
    batch_size = 4
    cls_pred = torch.randn(batch_size, 100, 2)
    bbox_pred = torch.randn(batch_size, 100, 4)
    kpt_pred = torch.randn(batch_size, 100, 10)
    
    bbox_target = torch.randn(batch_size, 100, 4)
    kpt_target = torch.randn(batch_size, 100, 10)
    
    loss, loss_info = criterion(cls_pred, bbox_pred, kpt_pred, bbox_target, kpt_target)
    print(f"Total Loss: {loss:.4f}")
    print(f"Classification Loss: {loss_info['cls_loss']:.4f}")
    print(f"BBox Loss: {loss_info['bbox_loss']:.4f}")
    print(f"Keypoint Loss: {loss_info['kpt_loss']:.4f}")
