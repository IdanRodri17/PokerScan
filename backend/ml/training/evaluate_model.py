"""
Model Evaluation and Metrics for PokerVision

This module provides comprehensive evaluation capabilities for trained
YOLOv8 poker card detection models.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import yaml
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)


class PokerModelEvaluator:
    """Comprehensive evaluation for poker card detection models"""
    
    def __init__(self, model_path: str, dataset_yaml_path: str, config_path: Optional[str] = None):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained model
            dataset_yaml_path: Path to dataset YAML
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path)
        self.dataset_yaml_path = Path(dataset_yaml_path)
        self.config_path = config_path
        
        # Load model and configuration
        self.model = None
        self.dataset_config = self._load_dataset_config()
        self.class_names = self._get_class_names()
        
        # Results storage
        self.evaluation_results = {}
        
        logger.info(f"Initialized evaluator for model: {model_path}")
    
    def _load_dataset_config(self) -> Dict:
        """Load dataset configuration"""
        try:
            with open(self.dataset_yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset config: {e}")
            return {}
    
    def _get_class_names(self) -> Dict[int, str]:
        """Get class ID to name mapping"""
        names = self.dataset_config.get('names', {})
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        else:
            return {}
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = YOLO(str(self.model_path))
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def evaluate_on_test_set(self) -> Dict:
        """Evaluate model on test set"""
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        logger.info("Evaluating model on test set...")
        
        try:
            # Run evaluation
            results = self.model.val(
                data=str(self.dataset_yaml_path),
                split='test',
                verbose=True,
                save_json=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(results.box.map50) if results.box.map50 is not None else 0.0,
                'mAP50_95': float(results.box.map) if results.box.map is not None else 0.0,
                'precision': float(results.box.p) if results.box.p is not None else 0.0,
                'recall': float(results.box.r) if results.box.r is not None else 0.0,
                'f1_score': float(results.box.f1) if results.box.f1 is not None else 0.0
            }
            
            # Per-class metrics
            if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
                per_class_ap50 = results.box.ap50
                if isinstance(per_class_ap50, torch.Tensor):
                    per_class_ap50 = per_class_ap50.cpu().numpy()
                
                metrics['per_class_ap50'] = {}
                for i, ap in enumerate(per_class_ap50):
                    class_name = self.class_names.get(i, f'class_{i}')
                    metrics['per_class_ap50'][class_name] = float(ap)
            
            self.evaluation_results['test_metrics'] = metrics
            logger.info(f"Test evaluation complete - mAP50: {metrics['mAP50']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Test evaluation failed: {e}")
            return {}
    
    def evaluate_inference_speed(self, num_samples: int = 100, image_size: Tuple[int, int] = (640, 640)) -> Dict:
        """
        Evaluate model inference speed
        
        Args:
            num_samples: Number of inference runs
            image_size: Input image size (width, height)
            
        Returns:
            Dict: Speed metrics
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        logger.info(f"Evaluating inference speed with {num_samples} samples...")
        
        # Create dummy images for speed testing
        dummy_image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
        
        inference_times = []
        
        try:
            # Warmup runs
            for _ in range(5):
                _ = self.model(dummy_image, verbose=False)
            
            # Timed runs
            for _ in range(num_samples):
                start_time = time.time()
                _ = self.model(dummy_image, verbose=False)
                inference_times.append((time.time() - start_time) * 1000)  # Convert to ms
            
            speed_metrics = {
                'mean_inference_time_ms': np.mean(inference_times),
                'median_inference_time_ms': np.median(inference_times),
                'min_inference_time_ms': np.min(inference_times),
                'max_inference_time_ms': np.max(inference_times),
                'std_inference_time_ms': np.std(inference_times),
                'fps': 1000.0 / np.mean(inference_times),
                'num_samples': num_samples
            }
            
            self.evaluation_results['speed_metrics'] = speed_metrics
            logger.info(f"Speed evaluation complete - Average: {speed_metrics['mean_inference_time_ms']:.1f}ms")
            
            return speed_metrics
            
        except Exception as e:
            logger.error(f"Speed evaluation failed: {e}")
            return {}
    
    def analyze_class_performance(self) -> Dict:
        """Analyze per-class performance"""
        test_metrics = self.evaluation_results.get('test_metrics', {})
        per_class_ap50 = test_metrics.get('per_class_ap50', {})
        
        if not per_class_ap50:
            logger.warning("No per-class metrics available")
            return {}
        
        # Analyze by suits and ranks
        suit_performance = {'spades': [], 'hearts': [], 'diamonds': [], 'clubs': []}
        rank_performance = {}
        
        for class_name, ap50 in per_class_ap50.items():
            if len(class_name) >= 2:
                rank = class_name[:-1]  # All but last character
                suit = class_name[-1]   # Last character
                
                # Group by suit
                suit_map = {'s': 'spades', 'h': 'hearts', 'd': 'diamonds', 'c': 'clubs'}
                if suit in suit_map:
                    suit_performance[suit_map[suit]].append(ap50)
                
                # Group by rank
                if rank not in rank_performance:
                    rank_performance[rank] = []
                rank_performance[rank].append(ap50)
        
        # Calculate averages
        analysis = {
            'suit_averages': {
                suit: np.mean(aps) if aps else 0.0 
                for suit, aps in suit_performance.items()
            },
            'rank_averages': {
                rank: np.mean(aps) if aps else 0.0 
                for rank, aps in rank_performance.items()
            },
            'best_classes': sorted(per_class_ap50.items(), key=lambda x: x[1], reverse=True)[:10],
            'worst_classes': sorted(per_class_ap50.items(), key=lambda x: x[1])[:10],
            'overall_class_std': np.std(list(per_class_ap50.values()))
        }
        
        self.evaluation_results['class_analysis'] = analysis
        return analysis
    
    def evaluate_on_custom_images(self, image_paths: List[str], confidence_threshold: float = 0.5) -> Dict:
        """
        Evaluate model on custom poker table images
        
        Args:
            image_paths: List of paths to test images
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Dict: Evaluation results
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        logger.info(f"Evaluating on {len(image_paths)} custom images...")
        
        results = {
            'images': [],
            'total_detections': 0,
            'average_confidence': 0.0,
            'detection_distribution': {}
        }
        
        all_confidences = []
        
        for img_path in image_paths:
            try:
                # Run inference
                detection_results = self.model(img_path, conf=confidence_threshold, verbose=False)
                
                if detection_results and len(detection_results) > 0:
                    result = detection_results[0]
                    
                    # Extract detection information
                    detections = []
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        for i in range(len(boxes)):
                            conf = float(boxes.conf[i].cpu().numpy())
                            cls_id = int(boxes.cls[i].cpu().numpy())
                            class_name = self.class_names.get(cls_id, f'unknown_{cls_id}')
                            
                            detections.append({
                                'class': class_name,
                                'confidence': conf
                            })
                            
                            all_confidences.append(conf)
                            
                            # Count detections by class
                            if class_name not in results['detection_distribution']:
                                results['detection_distribution'][class_name] = 0
                            results['detection_distribution'][class_name] += 1
                    
                    results['images'].append({
                        'path': img_path,
                        'detections': detections,
                        'num_detections': len(detections)
                    })
                    
                    results['total_detections'] += len(detections)
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
        
        # Calculate overall statistics
        if all_confidences:
            results['average_confidence'] = np.mean(all_confidences)
            results['confidence_std'] = np.std(all_confidences)
            results['min_confidence'] = np.min(all_confidences)
            results['max_confidence'] = np.max(all_confidences)
        
        self.evaluation_results['custom_images'] = results
        logger.info(f"Custom image evaluation complete - {results['total_detections']} total detections")
        
        return results
    
    def create_evaluation_plots(self, save_dir: str = "evaluation_plots") -> Dict[str, str]:
        """Create visualization plots for evaluation results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plots_created = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Class performance heatmap
            if 'class_analysis' in self.evaluation_results:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                class_analysis = self.evaluation_results['class_analysis']
                per_class_ap50 = self.evaluation_results.get('test_metrics', {}).get('per_class_ap50', {})
                
                if per_class_ap50:
                    # Create matrix for heatmap (13 ranks x 4 suits)
                    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
                    suits = ['s', 'h', 'd', 'c']
                    
                    matrix = np.zeros((len(ranks), len(suits)))
                    
                    for class_name, ap50 in per_class_ap50.items():
                        if len(class_name) >= 2:
                            rank = class_name[:-1]
                            suit = class_name[-1]
                            
                            if rank in ranks and suit in suits:
                                rank_idx = ranks.index(rank)
                                suit_idx = suits.index(suit)
                                matrix[rank_idx, suit_idx] = ap50
                    
                    sns.heatmap(matrix, 
                               xticklabels=['Spades', 'Hearts', 'Diamonds', 'Clubs'],
                               yticklabels=ranks,
                               annot=True, 
                               fmt='.2f',
                               cmap='RdYlBu_r',
                               ax=ax)
                    
                    ax.set_title('Per-Class AP50 Performance Heatmap')
                    ax.set_xlabel('Suit')
                    ax.set_ylabel('Rank')
                    
                    plot_path = save_dir / 'class_performance_heatmap.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plots_created['class_heatmap'] = str(plot_path)
                    plt.close()
            
            # 2. Speed performance plot
            if 'speed_metrics' in self.evaluation_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                speed_metrics = self.evaluation_results['speed_metrics']
                
                # Inference time distribution (if we had individual times)
                mean_time = speed_metrics['mean_inference_time_ms']
                std_time = speed_metrics['std_inference_time_ms']
                
                ax1.bar(['Mean Inference Time'], [mean_time], 
                       yerr=[std_time], capsize=10)
                ax1.set_ylabel('Time (ms)')
                ax1.set_title('Inference Speed Performance')
                
                # FPS comparison
                fps = speed_metrics['fps']
                target_fps = 1000.0 / 100.0  # 100ms target = 10 FPS
                
                ax2.bar(['Achieved', 'Target'], [fps, target_fps], 
                       color=['blue', 'red'], alpha=0.7)
                ax2.set_ylabel('Frames Per Second')
                ax2.set_title('FPS Performance vs Target')
                
                plt.tight_layout()
                plot_path = save_dir / 'speed_performance.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots_created['speed_performance'] = str(plot_path)
                plt.close()
            
            # 3. Overall metrics radar chart
            if 'test_metrics' in self.evaluation_results:
                test_metrics = self.evaluation_results['test_metrics']
                
                metrics = ['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1-Score']
                values = [
                    test_metrics.get('mAP50', 0),
                    test_metrics.get('mAP50_95', 0),
                    test_metrics.get('precision', 0),
                    test_metrics.get('recall', 0),
                    test_metrics.get('f1_score', 0)
                ]
                
                # Create radar chart
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                values = values + [values[0]]
                
                fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_ylim(0, 1)
                ax.set_title('Model Performance Metrics', size=16, pad=20)
                
                plot_path = save_dir / 'metrics_radar.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots_created['metrics_radar'] = str(plot_path)
                plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create plots: {e}")
        
        return plots_created
    
    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        report = {
            'model_path': str(self.model_path),
            'dataset_path': str(self.dataset_yaml_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_results': self.evaluation_results,
            'summary': {}
        }
        
        # Generate summary
        if 'test_metrics' in self.evaluation_results:
            test_metrics = self.evaluation_results['test_metrics']
            report['summary']['test_performance'] = {
                'mAP50': test_metrics.get('mAP50', 0),
                'meets_target': test_metrics.get('mAP50', 0) >= 0.90  # 90% target
            }
        
        if 'speed_metrics' in self.evaluation_results:
            speed_metrics = self.evaluation_results['speed_metrics']
            report['summary']['speed_performance'] = {
                'avg_inference_ms': speed_metrics.get('mean_inference_time_ms', 0),
                'meets_target': speed_metrics.get('mean_inference_time_ms', 1000) <= 100  # 100ms target
            }
        
        if 'class_analysis' in self.evaluation_results:
            class_analysis = self.evaluation_results['class_analysis']
            report['summary']['class_performance'] = {
                'best_suit': max(class_analysis['suit_averages'].items(), key=lambda x: x[1]),
                'worst_suit': min(class_analysis['suit_averages'].items(), key=lambda x: x[1]),
                'performance_variation': class_analysis['overall_class_std']
            }
        
        return report


def evaluate_model(model_path: str, dataset_yaml: str, custom_images: Optional[List[str]] = None) -> Dict:
    """
    Main evaluation function
    
    Args:
        model_path: Path to trained model
        dataset_yaml: Path to dataset YAML
        custom_images: Optional list of custom test images
        
    Returns:
        Dict: Comprehensive evaluation results
    """
    logger.info(f"Starting comprehensive model evaluation...")
    
    try:
        # Initialize evaluator
        evaluator = PokerModelEvaluator(model_path, dataset_yaml)
        
        if not evaluator.load_model():
            logger.error("Failed to load model")
            return {}
        
        # Run evaluations
        test_results = evaluator.evaluate_on_test_set()
        speed_results = evaluator.evaluate_inference_speed()
        class_analysis = evaluator.analyze_class_performance()
        
        # Evaluate on custom images if provided
        if custom_images:
            custom_results = evaluator.evaluate_on_custom_images(custom_images)
        
        # Create visualization plots
        plots = evaluator.create_evaluation_plots()
        
        # Generate final report
        report = evaluator.generate_evaluation_report()
        
        logger.info("Model evaluation completed successfully")
        return report
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return {}


if __name__ == "__main__":
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    model_path = "ml/models/poker_cards_best.pt"
    dataset_yaml = "data/processed/dataset.yaml"
    
    report = evaluate_model(model_path, dataset_yaml)
    
    if report:
        print("\\n🎯 Model evaluation completed!")
        print(f"Test mAP50: {report['evaluation_results']['test_metrics']['mAP50']:.3f}")
        print(f"Inference speed: {report['evaluation_results']['speed_metrics']['mean_inference_time_ms']:.1f}ms")
    else:
        print("\\n❌ Model evaluation failed.")