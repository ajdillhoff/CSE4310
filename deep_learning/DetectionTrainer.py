from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from torchvision.ops import box_iou
import numpy as np
from collections import defaultdict

from FasterRCNN import FasterRCNN

class DetectionTrainer:
    def __init__(self, model: FasterRCNN, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: optim.Optimizer, device: str = 'cuda', num_epochs: int = 12):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.best_map = 0.0
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for images, targets in tqdm(self.train_loader, desc='Training'):
            # Move data to device
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Get image shapes
            image_shapes = [(img.shape[2], img.shape[3]) for img in images]
            
            # Forward pass
            self.optimizer.zero_grad()
            losses = self.model(torch.stack(images), image_shapes, targets)
            loss = losses['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Evaluating'):
                # Move data to device
                images = [image.to(self.device) for image in images]
                image_shapes = [(img.shape[2], img.shape[3]) for img in images]
                
                # Get predictions
                outputs = self.model(torch.stack(images), image_shapes)
                
                # Convert outputs to list of predictions
                predictions = []
                for boxes, scores, labels in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
                    predictions.append({
                        'boxes': boxes.cpu(),
                        'scores': scores.cpu(),
                        'labels': labels.cpu()
                    })
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute mAP and other metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        return metrics

    def train(self) -> Dict[str, List[float]]:
        """Full training loop"""
        history = defaultdict(list)
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            
            # Train one epoch
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            
            # Evaluate
            metrics = self.evaluate()
            for k, v in metrics.items():
                history[k].append(v)
            
            # Print metrics
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Validation mAP@0.5: {metrics["mAP@0.5"]:.4f}')
            print(f'Validation mAP@0.5:0.95: {metrics["mAP@0.5:0.95"]:.4f}')
            
            # Save best model
            if metrics['mAP@0.5'] > self.best_map:
                self.best_map = metrics['mAP@0.5']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_map': self.best_map,
                }, 'best_model.pth')
        
        return history

    def compute_metrics(self, predictions: List[Dict[str, torch.Tensor]], 
                       targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Compute detection metrics including mAP"""
        
        def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
            """Compute IoU between two sets of boxes"""
            return box_iou(boxes1, boxes2)
        
        def compute_ap_per_class(pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                               pred_labels: torch.Tensor, true_boxes: torch.Tensor,
                               true_labels: torch.Tensor, iou_threshold: float) -> float:
            """Compute Average Precision for a single class"""
            if len(pred_boxes) == 0 or len(true_boxes) == 0:
                return 0.0
                
            # Compute IoUs between pred and true boxes
            iou_matrix = compute_iou_matrix(pred_boxes, true_boxes)
            
            # Sort predictions by score
            score_sort = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[score_sort]
            pred_scores = pred_scores[score_sort]
            pred_labels = pred_labels[score_sort]
            
            # Find matches
            true_matched = torch.zeros(len(true_boxes), dtype=torch.bool)
            pred_matched = torch.zeros(len(pred_boxes), dtype=torch.bool)
            
            for pred_idx in range(len(pred_boxes)):
                if pred_matched[pred_idx]:
                    continue
                    
                # Find true boxes for this class with IoU > threshold
                valid_true = (true_labels == pred_labels[pred_idx]) & (~true_matched)
                if not valid_true.any():
                    continue
                    
                # Find best matching true box
                ious = iou_matrix[pred_idx]
                best_true_idx = torch.argmax(ious * valid_true.float())
                if ious[best_true_idx] >= iou_threshold:
                    true_matched[best_true_idx] = True
                    pred_matched[pred_idx] = True
            
            # Compute precision and recall
            tp = torch.cumsum(pred_matched, dim=0)
            fp = torch.cumsum(~pred_matched, dim=0)
            
            # Avoid division by zero
            recall = tp / (len(true_boxes) + 1e-16)
            precision = tp / (tp + fp + 1e-16)
            
            # Compute average precision
            recall = torch.cat((torch.tensor([0]), recall, torch.tensor([1])))
            precision = torch.cat((torch.tensor([0]), precision, torch.tensor([0])))
            
            # Make precision monotonically decreasing
            for i in range(len(precision) - 2, -1, -1):
                precision[i] = max(precision[i], precision[i + 1])
            
            # Compute area under PR curve
            i = torch.where(recall[1:] != recall[:-1])[0]
            ap = torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])
            
            return ap.item()
        
        # Initialize metrics
        ap_per_class = defaultdict(list)
        iou_thresholds = torch.linspace(0.5, 0.95, 10)
        
        # Compute AP for each image and class
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            true_boxes = target['boxes']
            true_labels = target['labels']
            
            # Skip if no predictions or ground truth
            if len(pred_boxes) == 0 or len(true_boxes) == 0:
                continue
            
            # Compute AP for each class
            for label in torch.unique(torch.cat([pred_labels, true_labels])):
                pred_mask = pred_labels == label
                true_mask = true_labels == label
                
                if not pred_mask.any() or not true_mask.any():
                    continue
                
                # Compute AP at different IoU thresholds
                for iou_threshold in iou_thresholds:
                    ap = compute_ap_per_class(
                        pred_boxes[pred_mask], pred_scores[pred_mask],
                        pred_labels[pred_mask], true_boxes[true_mask],
                        true_labels[true_mask], iou_threshold.item()
                    )
                    ap_per_class[label.item()].append(ap)
        
        # Compute final metrics
        metrics = {}
        
        # mAP@0.5
        map50 = np.mean([aps[0] for aps in ap_per_class.values()])
        metrics['mAP@0.5'] = map50
        
        # mAP@0.5:0.95
        map = np.mean([np.mean(aps) for aps in ap_per_class.values()])
        metrics['mAP@0.5:0.95'] = map
        
        # Per-class AP@0.5
        for class_id, aps in ap_per_class.items():
            metrics[f'AP50_class_{class_id}'] = aps[0]
        
        return metrics

def create_trainer(model: FasterRCNN, train_loader: DataLoader, val_loader: DataLoader, 
                  learning_rate: float = 0.001, momentum: float = 0.9, 
                  weight_decay: float = 0.0005, **kwargs) -> DetectionTrainer:
    """Helper function to create a trainer with default settings"""
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    return DetectionTrainer(model, train_loader, val_loader, optimizer, **kwargs)

# Example usage
if __name__ == '__main__':
    # Initialize model, dataloaders, etc.
    model = FasterRCNN(n_classes=21)  # 20 classes + background
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=12
    )
    
    # Train model
    history = trainer.train()
    
    # Load best model for inference
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(images, image_shapes)