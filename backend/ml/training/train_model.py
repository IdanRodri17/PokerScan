"""
Complete YOLOv8 Training Pipeline for PokerVision

This module provides a comprehensive training pipeline for YOLOv8 models
specifically designed for poker card detection.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import torch
import numpy as np
from ultralytics import YOLO
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset_utils import download_and_process_dataset

logger = logging.getLogger(__name__)


class PokerCardTrainer:
    """YOLOv8 trainer for poker card detection"""
    
    def __init__(self, config_path: Optional[str] = None, use_wandb: bool = False):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
            use_wandb: Whether to use Weights & Biases for tracking
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.use_wandb = use_wandb
        
        # Training directories
        self.project_dir = Path("ml/training/runs")
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and training state
        self.model = None
        self.training_results = None
        
        logger.info("Initialized PokerCardTrainer")
    
    def _get_default_config_path(self) -> str:
        """Get default config path"""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "config" / "model_config.yaml")
    
    def _load_config(self) -> Dict:
        """Load training configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'model': {'name': 'yolov8n'},
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'patience': 50
            }
        }
    
    def prepare_dataset(self, force_download: bool = False) -> bool:
        """
        Prepare dataset for training
        
        Args:
            force_download: Force re-download of dataset
            
        Returns:
            bool: True if dataset preparation successful
        """
        logger.info("Preparing dataset for training...")
        
        dataset_dir = Path("data/processed")
        dataset_yaml = dataset_dir / "dataset.yaml"
        
        # Check if dataset already exists
        if dataset_yaml.exists() and not force_download:
            logger.info("Dataset already exists, skipping preparation")
            return True
        
        # Download and process dataset
        success = download_and_process_dataset(
            output_dir=str(dataset_dir),
            config=self.config.get('dataset', {})
        )
        
        if not success:
            logger.error("Dataset preparation failed")
            return False
        
        logger.info("Dataset preparation completed")
        return True
    
    def initialize_model(self, pretrained: bool = True) -> bool:
        """
        Initialize YOLOv8 model
        
        Args:
            pretrained: Use pretrained weights
            
        Returns:
            bool: True if model initialization successful
        """
        try:
            model_name = self.config.get('model', {}).get('name', 'yolov8n')
            
            if pretrained:
                model_path = f"{model_name}.pt"
                logger.info(f"Loading pretrained {model_name} model")
            else:
                model_path = f"{model_name}.yaml"
                logger.info(f"Loading {model_name} architecture only")
            
            self.model = YOLO(model_path)
            logger.info("Model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def setup_wandb(self, project_name: str = "pokervision-training") -> bool:
        """
        Setup Weights & Biases tracking
        
        Args:
            project_name: WandB project name
            
        Returns:
            bool: True if setup successful
        """
        if not self.use_wandb:
            return True
            
        try:
            wandb.init(
                project=project_name,
                config=self.config,
                name=f"poker_cards_{self.config.get('model', {}).get('name', 'yolov8n')}_{int(time.time())}"
            )
            logger.info("WandB tracking initialized")
            return True
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}")
            self.use_wandb = False
            return False
    
    def train(self, resume: Optional[str] = None) -> bool:
        """
        Train the model
        
        Args:
            resume: Path to checkpoint to resume from
            
        Returns:
            bool: True if training successful
        """
        if self.model is None:
            logger.error("Model not initialized. Call initialize_model() first.")
            return False
        
        logger.info("Starting model training...")
        
        try:
            # Get training parameters
            training_config = self.config.get('training', {})
            
            # Dataset path
            dataset_yaml = Path("data/processed/dataset.yaml")
            if not dataset_yaml.exists():
                logger.error("Dataset YAML not found. Run prepare_dataset() first.")
                return False
            
            # Training parameters
            epochs = training_config.get('epochs', 100)
            batch_size = training_config.get('batch_size', 16)
            learning_rate = training_config.get('learning_rate', 0.01)
            patience = training_config.get('patience', 50)
            
            # Start training
            results = self.model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                patience=patience,
                save_period=training_config.get('save_period', 10),
                project=str(self.project_dir),
                name=f"poker_cards_{int(time.time())}",
                exist_ok=True,
                verbose=True
            )
            
            self.training_results = results
            
            # Log to WandB if enabled
            if self.use_wandb:
                self._log_training_results(results)
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def validate(self) -> Dict:
        """
        Validate the trained model
        
        Returns:
            Dict: Validation metrics
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        logger.info("Running model validation...")
        
        try:
            # Run validation
            dataset_yaml = Path("data/processed/dataset.yaml")
            val_results = self.model.val(
                data=str(dataset_yaml),
                split='val',
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': val_results.box.map50,
                'mAP50-95': val_results.box.map,
                'precision': val_results.box.p,
                'recall': val_results.box.r,
                'f1': val_results.box.f1
            }
            
            logger.info(f"Validation complete - mAP50: {metrics['mAP50']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}
    
    def export_model(self, formats: List[str] = ['pt', 'onnx']) -> Dict[str, str]:
        """
        Export model to different formats
        
        Args:
            formats: List of export formats
            
        Returns:
            Dict mapping format to exported file path
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        logger.info(f"Exporting model to formats: {formats}")
        
        export_dir = Path("ml/models")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for format_type in formats:
            try:
                if format_type == 'pt':
                    # PyTorch format (already available)
                    pt_path = export_dir / "poker_cards_best.pt"
                    if hasattr(self.model, 'save'):
                        self.model.save(str(pt_path))
                    exported_files['pt'] = str(pt_path)
                    
                elif format_type == 'onnx':
                    # ONNX format for production
                    onnx_path = self.model.export(
                        format='onnx',
                        dynamic=True,
                        simplify=True
                    )
                    exported_files['onnx'] = onnx_path
                    
                elif format_type == 'torchscript':
                    # TorchScript format
                    ts_path = self.model.export(format='torchscript')
                    exported_files['torchscript'] = ts_path
                    
                else:
                    logger.warning(f"Unsupported export format: {format_type}")
                    
            except Exception as e:
                logger.error(f"Export to {format_type} failed: {e}")
        
        logger.info(f"Model exported to {len(exported_files)} formats")
        return exported_files
    
    def _log_training_results(self, results):
        """Log training results to WandB"""
        if not self.use_wandb:
            return
            
        try:
            # Log final metrics
            if hasattr(results, 'results_dict'):
                wandb.log(results.results_dict)
            
            # Log training curves
            if hasattr(results, 'curves'):
                wandb.log({"training_curves": wandb.Image(results.curves)})
                
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")
    
    def create_training_report(self) -> Dict:
        """Create comprehensive training report"""
        report = {
            'config': self.config,
            'model_info': self._get_model_info(),
            'dataset_info': self._get_dataset_info(),
            'training_completed': self.training_results is not None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if self.training_results:
            validation_metrics = self.validate()
            report['validation_metrics'] = validation_metrics
            
            # Performance analysis
            target_map50 = self.config.get('performance', {}).get('target_map50', 0.90)
            report['performance_analysis'] = {
                'target_map50': target_map50,
                'achieved_map50': validation_metrics.get('mAP50', 0),
                'target_met': validation_metrics.get('mAP50', 0) >= target_map50
            }
        
        return report
    
    def _get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'model_type': str(type(self.model)),
            'parameters': sum(p.numel() for p in self.model.parameters() if hasattr(self.model, 'parameters'))
        }
    
    def _get_dataset_info(self) -> Dict:
        """Get dataset information"""
        dataset_yaml = Path("data/processed/dataset.yaml")
        if not dataset_yaml.exists():
            return {'status': 'not_prepared'}
        
        try:
            with open(dataset_yaml, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            return {
                'status': 'prepared',
                'num_classes': dataset_config.get('nc', 0),
                'splits': ['train', 'val', 'test']
            }
        except Exception:
            return {'status': 'error_loading'}


def main_training_pipeline(config_path: Optional[str] = None, use_wandb: bool = False) -> bool:
    """
    Run the complete training pipeline
    
    Args:
        config_path: Path to configuration file
        use_wandb: Use Weights & Biases tracking
        
    Returns:
        bool: True if pipeline completed successfully
    """
    logger.info("Starting PokerVision training pipeline")
    
    try:
        # Initialize trainer
        trainer = PokerCardTrainer(config_path=config_path, use_wandb=use_wandb)
        
        # Setup tracking
        if use_wandb:
            trainer.setup_wandb()
        
        # Prepare dataset
        if not trainer.prepare_dataset():
            logger.error("Dataset preparation failed")
            return False
        
        # Initialize model
        if not trainer.initialize_model():
            logger.error("Model initialization failed")
            return False
        
        # Train model
        if not trainer.train():
            logger.error("Training failed")
            return False
        
        # Validate model
        validation_metrics = trainer.validate()
        logger.info(f"Validation metrics: {validation_metrics}")
        
        # Export model
        exported_models = trainer.export_model()
        logger.info(f"Exported models: {exported_models}")
        
        # Create training report
        report = trainer.create_training_report()
        logger.info("Training pipeline completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return False
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training pipeline
    success = main_training_pipeline(use_wandb=True)
    
    if success:
        print("\\n🎉 Training pipeline completed successfully!")
        print("Check ml/models/ for exported models")
    else:
        print("\\n❌ Training pipeline failed. Check logs for details.")
        exit(1)