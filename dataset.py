import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from pathlib import Path

class FaceDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, target_size=640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        self.image_dir = self.data_dir / split / 'images'
        self.annotation_dir = self.data_dir / split / 'annotations'
        
        self.image_files = sorted([f for f in self.image_dir.glob('*.jpg') if f.stem + '.xml' in [x.stem for x in self.annotation_dir.glob('*.xml')]])
        self.xml_files = {f.stem: f for f in self.annotation_dir.glob('*.xml')}
        
        self.keys = ['left_eye_center', 'right_eye_center', 'nose_tip', 
                     'left_mouth_corner', 'right_mouth_corner', 'face']
        
        print(f"Loaded {len(self.image_files)} images for {split} set")
        
    def __len__(self):
        return len(self.image_files)
    
    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        annotations = {}
        for key in self.keys:
            annotations[key] = None
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            if name in self.keys:
                annotations[name] = {
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                    'cx': (xmin + xmax) / 2,
                    'cy': (ymin + ymax) / 2
                }
        
        return width, height, annotations
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        xml_path = self.xml_files[img_path.stem]
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        width, height, annotations = self._parse_xml(xml_path)
        
        resized = cv2.resize(image, (self.target_size, self.target_size))
        resized = resized.astype(np.float32) / 255.0
        
        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h
        
        face_box = annotations['face']
        if face_box:
            bbox = np.array([
                face_box['xmin'] * scale_x,
                face_box['ymin'] * scale_y,
                face_box['xmax'] * scale_x,
                face_box['ymax'] * scale_y
            ], dtype=np.float32)
        else:
            bbox = np.zeros(4, dtype=np.float32)
        
        keypoints = np.zeros((5, 2), dtype=np.float32)
        keypoint_names = ['left_eye_center', 'right_eye_center', 'nose_tip', 
                          'left_mouth_corner', 'right_mouth_corner']
        
        for i, kp_name in enumerate(keypoint_names):
            kp = annotations[kp_name]
            if kp:
                keypoints[i, 0] = kp['cx'] * scale_x
                keypoints[i, 1] = kp['cy'] * scale_y
            else:
                keypoints[i, 0] = 0
                keypoints[i, 1] = 0
        
        image_tensor = torch.from_numpy(resized.transpose(2, 0, 1))
        bbox_tensor = torch.from_numpy(bbox)
        keypoints_tensor = torch.from_numpy(keypoints)
        
        return image_tensor, bbox_tensor, keypoints_tensor

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    bboxes = torch.stack([item[1] for item in batch])
    keypoints = torch.stack([item[2] for item in batch])
    return images, bboxes, keypoints

if __name__ == '__main__':
    dataset = FaceDataset(r'C:\Users\王子鸣\OneDrive\桌面\FaceDataset_organized', 'train')
    print(f"Dataset size: {len(dataset)}")
    
    img, bbox, kpts = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Bbox: {bbox}")
    print(f"Keypoints shape: {kpts.shape}")
