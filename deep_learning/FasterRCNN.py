import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from torchvision.models import resnet50
from typing import Dict, List, Tuple, Optional
import math

class AnchorGenerator:
    def __init__(self, scales: List[int], ratios: List[float], feature_strides: List[int]):
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides
        self.cell_anchors = self._generate_cell_anchors()

    def _generate_cell_anchors(self) -> List[torch.Tensor]:
        """Generate anchors for each feature map scale"""
        anchors = []
        for scale in self.scales:
            scale_anchors = []
            for ratio in self.ratios:
                w = scale * math.sqrt(ratio)
                h = scale / math.sqrt(ratio)
                
                # Center anchors at origin
                x0 = -w / 2
                y0 = -h / 2
                x1 = w / 2
                y1 = h / 2
                
                scale_anchors.append([x0, y0, x1, y1])
            anchors.append(torch.tensor(scale_anchors))
        return anchors

    def grid_anchors(self, grid_sizes: List[Tuple[int, int]], device: torch.device) -> List[torch.Tensor]:
        """Generate anchors for each feature map based on grid size"""
        anchors_over_all_feature_maps = []
        
        for size, stride, base_anchors in zip(grid_sizes, self.feature_strides, self.cell_anchors):
            grid_height, grid_width = size
            shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # Add anchors (4, A) to shifts (K, 4)
            base_anchors = base_anchors.to(device)
            anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            anchors_over_all_feature_maps.append(anchors)
            
        return anchors_over_all_feature_maps

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, scales: List[int], ratios: List[float], 
                 feature_strides: List[int]):
        super().__init__()
        self.anchor_generator = AnchorGenerator(scales, ratios, feature_strides)
        n_anchors = len(scales) * len(ratios)
        
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.cls_score = nn.Conv2d(mid_channels, n_anchors * 2, 1, 1, 0)
        self.bbox_pred = nn.Conv2d(mid_channels, n_anchors * 4, 1, 1, 0)
        
        # Initialize weights
        normal_init(self.conv, 0, 0.01)
        normal_init(self.cls_score, 0, 0.01)
        normal_init(self.bbox_pred, 0, 0.01)

    def forward(self, x: torch.Tensor, image_shapes: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        features = F.relu(self.conv(x))
        rpn_scores = self.cls_score(features)
        rpn_bbox_pred = self.bbox_pred(features)
        
        grid_sizes = [(math.ceil(image_shape[0] / stride), math.ceil(image_shape[1] / stride))
                     for image_shape, stride in zip(image_shapes, self.anchor_generator.feature_strides)]
        
        anchors = self.anchor_generator.grid_anchors(grid_sizes, x.device)
        return rpn_scores, rpn_bbox_pred, anchors

class RoIHead(nn.Module):
    def __init__(self, n_classes: int, roi_size: int = 7):
        super().__init__()
        # ResNet50 feature size
        in_channels = 2048
        
        self.avg_pool = nn.AdaptiveAvgPool2d(roi_size)
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        self.cls_score = nn.Linear(1024, n_classes)
        self.bbox_pred = nn.Linear(1024, n_classes * 4)
        
        # Initialize weights
        normal_init(self.fc1, 0, 0.01)
        normal_init(self.fc2, 0, 0.01)
        normal_init(self.cls_score, 0, 0.01)
        normal_init(self.bbox_pred, 0, 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        roi_scores = self.cls_score(x)
        roi_bboxes = self.bbox_pred(x)
        return roi_scores, roi_bboxes
    

class FasterRCNN(nn.Module):
    def __init__(self, n_classes: int, anchor_scales: List[int] = [8, 16, 32],
                 anchor_ratios: List[float] = [0.5, 1.0, 2.0],
                 feature_strides: List[int] = [16]):
        super().__init__()
        
        # Load pretrained ResNet50 as backbone
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Freeze early layers
        for layer in self.backbone[:5]:
            for p in layer.parameters():
                p.requires_grad = False
        
        self.rpn = RegionProposalNetwork(2048, 512, anchor_scales, anchor_ratios, feature_strides)
        self.roi_head = RoIHead(n_classes)
        
        self.n_classes = n_classes
        self.min_size = 16
        self.nms_thresh = 0.7
        self.post_nms_top_n = 1000
        self.rpn_pre_nms_top_n = 2000
        self.rpn_post_nms_top_n = 2000
        self.rpn_nms_thresh = 0.7
        self.rpn_min_size = 16

    def forward(self, images: torch.Tensor, image_shapes: List[Tuple[int, int]], 
                targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        
        # RPN forward pass
        rpn_scores, rpn_bbox_pred, anchors = self.rpn(features, image_shapes)
        
        if self.training:
            assert targets is not None
            rpn_loss = self.compute_rpn_loss(rpn_scores, rpn_bbox_pred, anchors, targets)
            
            # Generate proposals from RPN predictions
            proposals = self.generate_proposals(rpn_bbox_pred, rpn_scores, anchors, image_shapes)
            
            # ROI pooling and head forward pass
            pooled_features = self.roi_pooling(features, proposals)
            roi_scores, roi_bboxes = self.roi_head(pooled_features)
            
            roi_loss = self.compute_roi_loss(roi_scores, roi_bboxes, proposals, targets)
            
            return {
                'rpn_loss': rpn_loss,
                'roi_loss': roi_loss,
                'total_loss': rpn_loss + roi_loss
            }
        else:
            proposals = self.generate_proposals(rpn_bbox_pred, rpn_scores, anchors, image_shapes)
            pooled_features = self.roi_pooling(features, proposals)
            roi_scores, roi_bboxes = self.roi_head(pooled_features)
            
            # Post-processing
            final_boxes, final_scores, final_labels = self.postprocess_detections(
                roi_bboxes, roi_scores, proposals, image_shapes)
                
            return {
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            }

    def generate_proposals(self, bbox_pred: torch.Tensor, scores: torch.Tensor,
                         anchors: List[torch.Tensor], image_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """Generate RPN proposals for each image"""
        proposals = []
        for bbox_pred_per_image, scores_per_image, anchors_per_image, image_shape in zip(
            bbox_pred, scores, anchors, image_shapes):
            
            # Convert RPN box predictions to absolute coordinates
            proposals_per_image = self.apply_deltas_to_anchors(
                bbox_pred_per_image.reshape(-1, 4),
                anchors_per_image
            )
            
            # Clip to image boundaries
            proposals_per_image = self.clip_boxes_to_image(proposals_per_image, image_shape)
            
            # Remove small boxes
            keep = self.remove_small_boxes(proposals_per_image, self.rpn_min_size)
            proposals_per_image = proposals_per_image[keep]
            scores_per_image = scores_per_image[keep]
            
            # Sort by score and apply NMS
            scores_per_image = scores_per_image.reshape(-1)
            sorted_scores, order = torch.sort(scores_per_image, descending=True)
            order = order[:self.rpn_pre_nms_top_n]
            proposals_per_image = proposals_per_image[order]
            scores_per_image = sorted_scores[:self.rpn_pre_nms_top_n]
            
            keep = nms(proposals_per_image, scores_per_image, self.rpn_nms_thresh)
            keep = keep[:self.rpn_post_nms_top_n]
            proposals_per_image = proposals_per_image[keep]
            
            proposals.append(proposals_per_image)
        
        return proposals

    def roi_pooling(self, features: torch.Tensor, proposals: List[torch.Tensor]) -> torch.Tensor:
        """Apply ROI pooling to features using proposal boxes"""
        from torchvision.ops import roi_align
        
        # Concatenate proposals and add batch indices
        proposal_boxes = []
        for i, boxes in enumerate(proposals):
            batch_idx = torch.full((len(boxes), 1), i, dtype=torch.float32, device=boxes.device)
            proposal_boxes.append(torch.cat([batch_idx, boxes], dim=1))
        proposal_boxes = torch.cat(proposal_boxes, dim=0)
        
        # Apply ROI align
        return roi_align(features, proposal_boxes, output_size=(7, 7), spatial_scale=1.0/16.0)

    def postprocess_detections(self, box_regression: torch.Tensor, class_logits: torch.Tensor,
                             proposals: List[torch.Tensor], image_shapes: List[Tuple[int, int]]) -> Tuple[
                                 List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Post-process detection outputs"""
        device = box_regression.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.apply_deltas_to_proposals(box_regression, proposals)
        
        # Split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores = F.softmax(class_logits, -1)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = self.clip_boxes_to_image(boxes, image_shape)
            
            # Create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            # Remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
            # Flatten boxes and scores
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            
            # Remove low scoring boxes
            inds = torch.where(scores > 0.05)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            
            # Remove empty boxes
            keep = self.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            # Apply NMS for each class independently
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.post_nms_top_n]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        
        return all_boxes, all_scores, all_labels

    @staticmethod
    def clip_boxes_to_image(boxes: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        """Clip boxes to image boundaries"""
        dim = boxes.dim()
        boxes_x = boxes[..., 0::2]
        boxes_y = boxes[..., 1::2]
        height, width = image_shape
        
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)
        
        clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
        return clipped_boxes.reshape(boxes.shape)

    @staticmethod
    def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
        """Remove boxes with any side smaller than min_size"""
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        return torch.where(keep)[0]

    def apply_deltas_to_anchors(self, deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Apply regression deltas to anchors"""
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        # Prevent sending too large values into exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def apply_deltas_to_proposals(self, deltas: torch.Tensor, proposals: List[torch.Tensor]) -> torch.Tensor:
        """Apply regression deltas to proposals"""
        proposals = torch.cat(proposals, dim=0)
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        ctr_x = proposals[:, 0] + 0.5 * widths
        ctr_y = proposals[:, 1] + 0.5 * heights

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = deltas.new_zeros(deltas.shape)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

def normal_init(module: nn.Module, mean: float, std: float):
    """Initialize network weights with normal distribution"""
    module.weight.data.normal_(mean, std)
    if module.bias is not None:
        module.bias.data.zero_()