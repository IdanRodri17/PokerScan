"""
Dataset Utilities for PokerVision YOLOv8 Training

This module handles dataset downloading, preprocessing, and preparation
for YOLOv8 training with the Kaggle Playing Cards dataset.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import zipfile
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KaggleDatasetDownloader:
    """Download and prepare Kaggle datasets"""
    
    def __init__(self, dataset_name: str = "andy8744/playing-cards-object-detection-dataset"):
        """
        Initialize dataset downloader
        
        Args:
            dataset_name: Kaggle dataset identifier
        """
        self.dataset_name = dataset_name
        self.download_dir = Path("data/raw")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> bool:
        """
        Download dataset from Kaggle using kaggle API
        
        Returns:
            bool: True if download successful
        """
        try:
            # Check if kaggle is available
            import kaggle
            
            logger.info(f"Downloading {self.dataset_name} from Kaggle...")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=str(self.download_dir),
                unzip=True
            )
            
            logger.info("Dataset downloaded successfully")
            return True
            
        except ImportError:
            logger.error("Kaggle API not available. Install with: pip install kaggle")
            logger.info("Alternative: Download manually from https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset")
            return False
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False


class PokerCardDatasetProcessor:
    """Process poker card dataset for YOLOv8 training"""
    
    def __init__(self, raw_data_dir: str, output_dir: str, config: Optional[Dict] = None):
        """
        Initialize dataset processor
        
        Args:
            raw_data_dir: Directory containing raw dataset
            output_dir: Directory for processed dataset
            config: Dataset configuration
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset splits
        self.splits = ['train', 'val', 'test']
        for split in self.splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self, train_split: float = 0.8, val_split: float = 0.15, test_split: float = 0.05) -> bool:
        """
        Process the raw dataset into YOLOv8 format
        
        Args:
            train_split: Fraction for training set
            val_split: Fraction for validation set
            test_split: Fraction for test set
            
        Returns:
            bool: True if processing successful
        """
        try:
            logger.info("Processing poker card dataset...")
            
            # Find and process annotation files
            annotation_files = list(self.raw_data_dir.glob("*.csv"))
            if not annotation_files:
                annotation_files = list(self.raw_data_dir.glob("**/*.csv"))
            
            if not annotation_files:
                logger.error("No annotation files found")
                return False
            
            # Process each annotation file
            all_data = []
            for ann_file in annotation_files:
                data = self._process_annotation_file(ann_file)
                all_data.extend(data)
            
            if not all_data:
                logger.error("No valid annotations processed")
                return False
            
            # Split dataset
            train_data, temp_data = train_test_split(
                all_data, 
                test_size=(val_split + test_split),
                random_state=42
            )
            
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(test_split / (val_split + test_split)),
                random_state=42
            )
            
            # Process each split
            splits_data = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            for split_name, split_data in splits_data.items():
                self._process_split(split_name, split_data)
            
            # Create dataset YAML file
            self._create_dataset_yaml()
            
            logger.info(f"Dataset processed successfully: {len(train_data)} train, "
                       f"{len(val_data)} val, {len(test_data)} test samples")
            return True
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            return False
    
    def _process_annotation_file(self, annotation_file: Path) -> List[Dict]:
        """Process a single annotation file"""
        try:
            df = pd.read_csv(annotation_file)
            logger.info(f"Processing {annotation_file} with {len(df)} annotations")
            
            # Group by image
            processed_data = []
            for image_name, group in df.groupby('filename'):
                image_path = self._find_image_file(image_name)
                if image_path:
                    annotations = []
                    for _, row in group.iterrows():
                        annotation = self._convert_annotation(row)
                        if annotation:
                            annotations.append(annotation)
                    
                    if annotations:
                        processed_data.append({
                            'image_path': image_path,
                            'image_name': image_name,
                            'annotations': annotations
                        })
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to process annotation file {annotation_file}: {e}")
            return []
    
    def _find_image_file(self, image_name: str) -> Optional[Path]:
        """Find image file in raw data directory"""
        # Common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in extensions:
            # Try direct path
            image_path = self.raw_data_dir / f"{Path(image_name).stem}{ext}"
            if image_path.exists():
                return image_path
            
            # Try in subdirectories
            for subdir in self.raw_data_dir.iterdir():
                if subdir.is_dir():
                    image_path = subdir / f"{Path(image_name).stem}{ext}"
                    if image_path.exists():
                        return image_path
        
        logger.warning(f"Image not found: {image_name}")
        return None
    
    def _convert_annotation(self, row: pd.Series) -> Optional[Dict]:
        """Convert annotation row to YOLO format"""
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            width, height = row['width'], row['height']
            
            # Convert to YOLO format (normalized center coordinates and dimensions)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # Get class name and convert to class ID
            class_name = row['class']
            class_id = self._get_class_id(class_name)
            
            return {
                'class_id': class_id,
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': bbox_width,
                'height': bbox_height
            }
            
        except Exception as e:
            logger.warning(f"Failed to convert annotation: {e}")
            return None
    
    def _get_class_id(self, class_name: str) -> int:
        """Convert class name to class ID"""
        # Standard 52-card deck mapping
        card_to_id = {
            # Spades
            'ace of spades': 0, '2 of spades': 1, '3 of spades': 2, '4 of spades': 3,
            '5 of spades': 4, '6 of spades': 5, '7 of spades': 6, '8 of spades': 7,
            '9 of spades': 8, '10 of spades': 9, 'jack of spades': 10, 'queen of spades': 11, 'king of spades': 12,
            
            # Hearts
            'ace of hearts': 13, '2 of hearts': 14, '3 of hearts': 15, '4 of hearts': 16,
            '5 of hearts': 17, '6 of hearts': 18, '7 of hearts': 19, '8 of hearts': 20,
            '9 of hearts': 21, '10 of hearts': 22, 'jack of hearts': 23, 'queen of hearts': 24, 'king of hearts': 25,
            
            # Diamonds
            'ace of diamonds': 26, '2 of diamonds': 27, '3 of diamonds': 28, '4 of diamonds': 29,
            '5 of diamonds': 30, '6 of diamonds': 31, '7 of diamonds': 32, '8 of diamonds': 33,
            '9 of diamonds': 34, '10 of diamonds': 35, 'jack of diamonds': 36, 'queen of diamonds': 37, 'king of diamonds': 38,
            
            # Clubs
            'ace of clubs': 39, '2 of clubs': 40, '3 of clubs': 41, '4 of clubs': 42,
            '5 of clubs': 43, '6 of clubs': 44, '7 of clubs': 45, '8 of clubs': 46,
            '9 of clubs': 47, '10 of clubs': 48, 'jack of clubs': 49, 'queen of clubs': 50, 'king of clubs': 51
        }
        
        # Normalize class name
        normalized_name = class_name.lower().strip()
        return card_to_id.get(normalized_name, 0)  # Default to Ace of Spades if not found
    
    def _process_split(self, split_name: str, split_data: List[Dict]):
        """Process a single dataset split"""
        logger.info(f"Processing {split_name} split with {len(split_data)} images")
        
        for i, data in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            try:
                image_path = data['image_path']
                annotations = data['annotations']
                
                # Copy image to split directory
                new_image_name = f"{split_name}_{i:06d}{image_path.suffix}"
                new_image_path = self.output_dir / split_name / 'images' / new_image_name
                shutil.copy2(image_path, new_image_path)
                
                # Create label file
                label_name = f"{split_name}_{i:06d}.txt"
                label_path = self.output_dir / split_name / 'labels' / label_name
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
                
            except Exception as e:
                logger.error(f"Failed to process {data.get('image_name', 'unknown')}: {e}")
    
    def _create_dataset_yaml(self):
        """Create dataset YAML file for YOLOv8"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 52,  # Number of classes
            'names': {
                # Spades
                0: "As", 1: "2s", 2: "3s", 3: "4s", 4: "5s", 5: "6s", 6: "7s", 7: "8s",
                8: "9s", 9: "Ts", 10: "Js", 11: "Qs", 12: "Ks",
                # Hearts
                13: "Ah", 14: "2h", 15: "3h", 16: "4h", 17: "5h", 18: "6h", 19: "7h", 20: "8h",
                21: "9h", 22: "Th", 23: "Jh", 24: "Qh", 25: "Kh",
                # Diamonds
                26: "Ad", 27: "2d", 28: "3d", 29: "4d", 30: "5d", 31: "6d", 32: "7d", 33: "8d",
                34: "9d", 35: "Td", 36: "Jd", 37: "Qd", 38: "Kd",
                # Clubs
                39: "Ac", 40: "2c", 41: "3c", 42: "4c", 43: "5c", 44: "6c", 45: "7c", 46: "8c",
                47: "9c", 48: "Tc", 49: "Jc", 50: "Qc", 51: "Kc"
            }
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logger.info(f"Created dataset YAML: {yaml_path}")
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the processed dataset"""
        stats = {'splits': {}}
        
        for split in self.splits:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            
            num_images = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
            num_labels = len(list(labels_dir.glob('*'))) if labels_dir.exists() else 0
            
            # Count annotations per class
            class_counts = {}
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.strip().split()[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            stats['splits'][split] = {
                'num_images': num_images,
                'num_labels': num_labels,
                'class_distribution': class_counts,
                'total_annotations': sum(class_counts.values())
            }
        
        return stats


def download_and_process_dataset(output_dir: str = "data/processed", config: Optional[Dict] = None) -> bool:
    """
    Main function to download and process the Kaggle dataset
    
    Args:
        output_dir: Directory for processed dataset
        config: Configuration dictionary
        
    Returns:
        bool: True if successful
    """
    try:
        # Download dataset
        downloader = KaggleDatasetDownloader()
        if not downloader.download_dataset():
            logger.warning("Dataset download failed, assuming data exists locally")
        
        # Process dataset
        processor = PokerCardDatasetProcessor(
            raw_data_dir="data/raw",
            output_dir=output_dir,
            config=config
        )
        
        success = processor.process_dataset()
        
        if success:
            stats = processor.get_dataset_statistics()
            logger.info("Dataset processing completed successfully")
            logger.info(f"Dataset statistics: {stats}")
        
        return success
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return False