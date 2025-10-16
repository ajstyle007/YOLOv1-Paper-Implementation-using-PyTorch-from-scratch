import torch
from torch import nn
import numpy as np

class YOLOv1(nn.Module):
    def __init__(self, S=4, B=2, C=20, backbone='resnet18', pretrained=True):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        layers = list(base_model.children())[:-2]  # remove avgpool & fc layer
        self.backbone = nn.Sequential(*layers)
        out_channels = 512  # resnet18 last conv output

        # ===== 2️⃣ DETECTION HEAD =====
        # The head converts [Batch, out_channels, feature_h, feature_w]
        # to [Batch, S, S, C + 5B]

        self.conv_head = nn.Sequential(
            nn.Conv2d(out_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, (C + 5 * B), kernel_size=1)  # final prediction map
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_head(x)

        # Reshape output -> [N, S, S, C + 5B]
        # (feature map size should match SxS)
        x = nn.functional.adaptive_avg_pool2d(x, (self.S, self.S))
        x = x.permute(0, 2, 3, 1)  # [N, S, S, C + 5B]
        return x
    


import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv1(S=4, B=2, C=20)  # initialize model

# device = "cpu"
# load checkpoint
checkpoint = torch.load("model/best_model.pth", map_location=device)

# load only the model weights
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()  # important for inference



def cellboxes_to_boxes(target, S=4, B=2, C=20, img_size=(224, 224)):
    """
    Convert target tensor [S, S, C+5*B] to list of boxes [x1, y1, x2, y2, conf, class_idx].
    """
    boxes = []
    H, W = img_size

    for i in range(S):
        for j in range(S):
            cell = target[i, j]
            for b in range(B):
                conf = cell[5*b + 4].item()
                if conf == 0:  # Skip cells without objects
                    continue
                x_cell = cell[5*b].item()
                y_cell = cell[5*b + 1].item()
                w = cell[5*b + 2].item()
                h = cell[5*b + 3].item()

                # Validate coordinates
                if not (0 <= x_cell <= 1 and 0 <= y_cell <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    warnings.warn(f"Invalid GT box coordinates: [x_cell={x_cell}, y_cell={y_cell}, w={w}, h={h}] at grid ({i}, {j}), box {b}")
                    continue

                # Cell to image scale (normalized [0, 1])
                x_center = (j + x_cell) / S
                y_center = (i + y_cell) / S

                # Convert to pixel coordinates
                x1 = (x_center - w/2) * W
                y1 = (y_center - h/2) * H
                x2 = (x_center + w/2) * W
                y2 = (y_center + h/2) * H

                # Clip boxes
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                # Validate box dimensions
                if x2 <= x1 or y2 <= y1:
                    warnings.warn(f"Invalid GT box dimensions: [x1={x1}, y1={y1}, x2={x2}, y2={y2}] at grid ({i}, {j}), box {b}")
                    continue

                # Class prediction
                class_probs = cell[5*B:]
                class_idx = torch.argmax(class_probs).item()

                boxes.append([x1, y1, x2, y2, conf, class_idx])

    return boxes


import numpy as np

def intersection_over_union(boxes_preds, boxes_labels):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format
    Handles both tensor and numpy/list inputs
    """
    # Convert to tensors if needed
    if isinstance(boxes_preds, (list, np.ndarray)):
        boxes_preds = torch.tensor(boxes_preds, dtype=torch.float32)
    if isinstance(boxes_labels, (list, np.ndarray)):
        boxes_labels = torch.tensor(boxes_labels, dtype=torch.float32)
    
    # Extract coordinates
    box1 = boxes_preds[:4]  # x1, y1, x2, y2
    box2 = boxes_labels[:4]
    
    # Intersection coordinates
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    
    # Intersection area
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Handle edge cases
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    return iou.item() if iou.numel() == 1 else iou



from collections import defaultdict

def scale_boxes(boxes, size=224):
    """
    Scale normalized boxes [img_idx, cls, conf, x_center, y_center, w, h] to pixel coordinates.
    Returns: [img_idx, cls, conf, x1, y1, x2, y2]
    """
    scaled = []
    for b in boxes:
        if len(b) != 7:
            print(f"Warning: Invalid box format: {b}")
            continue
        img_idx, cls, conf, x_center, y_center, w, h = b
        
        # Validate normalized coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            print(f"Warning: Invalid normalized coordinates in box: {b}")
            continue
        
        # Convert to pixel coordinates
        x1 = (x_center - w/2) * size
        y1 = (y_center - h/2) * size
        x2 = (x_center + w/2) * size
        y2 = (y_center + h/2) * size
        
        # Clip coordinates to image boundaries
        x1 = max(0, min(x1, size))
        y1 = max(0, min(y1, size))
        x2 = max(0, min(x2, size))
        y2 = max(0, min(y2, size))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid box dimensions: {b}")
            continue
            
        scaled.append([img_idx, cls, conf, x1, y1, x2, y2])
    return scaled



def convert_gt_boxes(gt_boxes, size=224):
    """
    Convert ground truth boxes from [img_idx, cls, x_center, y_center, w, h] normalized
    to [img_idx, cls, x1, y1, x2, y2] in pixel coordinates.
    """
    converted = []
    for b in gt_boxes:
        if len(b) != 6:
            print(f"Warning: Invalid ground truth box format: {b}")
            continue
        img_idx, cls, x_center, y_center, w, h = b
        
        # Validate normalized coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            print(f"Warning: Invalid normalized coordinates in GT box: {b}")
            continue
        
        # Convert to pixel coordinates
        x1 = (x_center - w/2) * size
        y1 = (y_center - h/2) * size
        x2 = (x_center + w/2) * size
        y2 = (y_center + h/2) * size
        
        # Clip coordinates to image boundaries
        x1 = max(0, min(x1, size))
        y1 = max(0, min(y1, size))
        x2 = max(0, min(x2, size))
        y2 = max(0, min(y2, size))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid GT box dimensions: {b}")
            continue
            
        converted.append([img_idx, cls, x1, y1, x2, y2])
    return converted

def intersection_over_union(boxes_preds, boxes_labels):
    """
    Compute IoU for boxes in [x1, y1, x2, y2] format.
    Handles both list and tensor inputs.
    """
    # Convert inputs to tensors
    if not isinstance(boxes_preds, torch.Tensor):
        boxes_preds = torch.tensor(boxes_preds, dtype=torch.float32)
    if not isinstance(boxes_labels, torch.Tensor):
        boxes_labels = torch.tensor(boxes_labels, dtype=torch.float32)
    
    # Extract coordinates
    box1_x1, box1_y1, box1_x2, box1_y2 = boxes_preds
    box2_x1, box2_y1, box2_x2, box2_y2 = boxes_labels
    
    # Compute intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # Intersection area
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union area
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    # Compute IoU
    iou = inter / (box1_area + box2_area - inter + 1e-6)
    
    return iou.item()



def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20, size=224, class_only=False):
    """
    Compute mAP or class-only accuracy.
    pred_boxes: list of [img_idx, cls, conf, x_center, y_center, w, h] normalized
    true_boxes: list of [img_idx, cls, x_center, y_center, w, h] normalized
    """
    # Validate inputs
    if not pred_boxes or not true_boxes:
        print(f"Warning: Empty box lists - pred_boxes: {len(pred_boxes)}, true_boxes: {len(true_boxes)}")
        return 0.0 if not class_only else 0.0

    # Convert boxes to pixel coordinates
    true_boxes = convert_gt_boxes(true_boxes, size=size)
    pred_boxes = scale_boxes(pred_boxes, size=size)

    if class_only:
        # Simple class accuracy
        correct_classes = 0
        total_gt = len(true_boxes)
        gt_per_image = defaultdict(list)
        for gt in true_boxes:
            gt_per_image[gt[0]].append(gt)

        for pred in pred_boxes:
            img_idx = pred[0]
            pred_cls = pred[1]  # cls is at index 1
            matched = False
            for gt in gt_per_image[img_idx]:
                if gt[1] == pred_cls:
                    correct_classes += 1
                    gt_per_image[img_idx].remove(gt)
                    matched = True
                    break
            if not matched and len(pred_boxes) < 10:  # Limit debug output
                print(f"Unmatched prediction: img_idx={img_idx}, cls={pred_cls}")
        class_acc = correct_classes / max(total_gt, 1)
        print(f"Class-only accuracy: {class_acc:.4f}, Correct: {correct_classes}/{total_gt}")
        return class_acc

    # Standard mAP computation
    average_precisions = []

    for c in range(num_classes):
        detections = [d for d in pred_boxes if d[1] == c]  # cls at index 1
        ground_truths = [t for t in true_boxes if t[1] == c]

        amount_bboxes = defaultdict(int)
        for gt in ground_truths:
            amount_bboxes[gt[0]] += 1

        for key in amount_bboxes.keys():
            amount_bboxes[key] = torch.zeros(amount_bboxes[key])

        detections.sort(key=lambda x: x[2], reverse=True)  # sort by conf (index 2)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            print(f"No ground truth boxes for class {c}")
            continue

        for det_idx, detection in enumerate(detections):
            img_idx = detection[0]
            best_iou = 0
            best_gt_idx = -1

            gt_for_img = [gt for gt in ground_truths if gt[0] == img_idx]
            for idx, gt in enumerate(gt_for_img):
                iou = intersection_over_union(detection[3:7], gt[2:6])  # [x1, y1, x2, y2]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if best_gt_idx >= 0 and amount_bboxes[img_idx][best_gt_idx] == 0:
                    TP[det_idx] = 1
                    amount_bboxes[img_idx][best_gt_idx] = 1
                else:
                    FP[det_idx] = 1
            else:
                FP[det_idx] = 1

            if det_idx < 5:  # Debug first few IoUs
                print(f"Class {c}, Detection {det_idx}: Best IoU={best_iou:.4f}, TP={TP[det_idx]}, FP={FP[det_idx]}")

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        precisions = torch.cat((torch.tensor([1.0]), precisions))
        recalls = torch.cat((torch.tensor([0.0]), recalls))
        ap = torch.trapz(precisions, recalls)
        average_precisions.append(ap)
        print(f"Class {c}: AP={ap:.4f}, TP={TP.sum().item()}, FP={FP.sum().item()}, GT={total_true_bboxes}")

    if len(average_precisions) == 0:
        print("No valid classes for mAP calculation")
        return 0.0
    
    mAP = sum(average_precisions) / len(average_precisions)
    print(f"mAP@{iou_threshold}: {mAP:.4f}, Valid classes: {len(average_precisions)}/{num_classes}")
    return mAP



import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import warnings

class YOLOdataset(Dataset):
    def __init__(self, image_dir, annot_dir, S=4, B=2, C=20, transform=None, target_size=(224, 224)):
        """
        Initialize the YOLO dataset.
        Args:
            image_dir (str): Directory containing images.
            annot_dir (str): Directory containing VOC XML annotations.
            S (int): Grid size (S x S).
            B (int): Number of bounding boxes per grid cell.
            C (int): Number of classes.
            transform: torchvision transforms for images.
            target_size (tuple): Size to normalize coordinates to (default: 224x224).
        """
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.S, self.B, self.C = S, B, C
        self.transform = transform
        self.target_size = target_size

        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

        # Validate directories
        if not os.path.exists(image_dir) or not os.path.exists(annot_dir):
            raise FileNotFoundError(f"Image dir {image_dir} or annotation dir {annot_dir} does not exist")
        if not self.image_files:
            raise ValueError(f"No valid image files found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_filename)
        annot_path = os.path.join(self.annot_dir, img_filename.rsplit(".", 1)[0] + ".xml")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            return None, None

        try:
            boxes, labels = self.parse_voc_xml(annot_path)
            target = self.encode_target(boxes, labels)
        except Exception as e:
            warnings.warn(f"Failed to parse annotation {annot_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, target

    def parse_voc_xml(self, xml_path):
        """
        Parse VOC XML annotation file and return normalized boxes and labels.
        Returns: boxes ([x_center, y_center, w, h] normalized to target_size), labels (class indices)
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Annotation file {xml_path} not found")

        boxes, labels = [], []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        w, h = self.target_size  # Normalize to target size (224x224)

        # Get original image size from XML
        size = root.find("size")
        orig_w = float(size.find("width").text)
        orig_h = float(size.find("height").text)

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.classes:
                warnings.warn(f"Unknown class {label} in {xml_path}, skipping object")
                continue

            xml_box = obj.find("bndbox")
            try:
                xmin = float(xml_box.find("xmin").text)
                ymin = float(xml_box.find("ymin").text)
                xmax = float(xml_box.find("xmax").text)
                ymax = float(xml_box.find("ymax").text)
            except ValueError as e:
                warnings.warn(f"Invalid bounding box coordinates in {xml_path}: {e}")
                continue

            # Validate box coordinates
            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                warnings.warn(f"Invalid box [xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}] in {xml_path}")
                continue

            # Convert to normalized coordinates relative to original size
            x_center = ((xmin + xmax) / 2) / orig_w
            y_center = ((ymin + ymax) / 2) / orig_h
            box_w = (xmax - xmin) / orig_w
            box_h = (ymax - ymin) / orig_h

            # Scale to target size (224x224)
            x_center = x_center * (orig_w / w)
            y_center = y_center * (orig_h / h)
            box_w = box_w * (orig_w / w)
            box_h = box_h * (orig_h / h)

            # Validate normalized coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= box_w <= 1 and 0 <= box_h <= 1):
                warnings.warn(f"Invalid normalized box [x={x_center}, y={y_center}, w={box_w}, h={box_h}] in {xml_path}")
                continue

            boxes.append([x_center, y_center, box_w, box_h])
            labels.append(self.classes.index(label))

        if not boxes:
            warnings.warn(f"No valid objects found in {xml_path}")

        return boxes, labels

    def encode_target(self, boxes, labels):
        """
        Encode ground truth boxes and labels into target tensor [S, S, C + 5*B].
        Assigns boxes to grid cells, using one box per cell (best IoU if multiple).
        """
        S, B, C = self.S, self.B, self.C
        target = torch.zeros((S, S, C + 5 * B))

        # Group boxes by grid cell
        cell_boxes = defaultdict(list)
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            grid_x = min(int(S * x), S - 1)
            grid_y = min(int(S * y), S - 1)
            cell_boxes[(grid_y, grid_x)].append((box, label))

        # Assign one box per cell (use first box or best IoU if needed)
        for (grid_y, grid_x), box_list in cell_boxes.items():
            # For simplicity, use the first box (could extend to best IoU)
            box, label = box_list[0]  # TODO: Add IoU-based selection for multiple boxes
            x, y, w, h = box
            x_cell = S * x - grid_x
            y_cell = S * y - grid_y

            # Validate cell coordinates
            if not (0 <= x_cell <= 1 and 0 <= y_cell <= 1):
                warnings.warn(f"Invalid cell coordinates [x_cell={x_cell}, y_cell={y_cell}]")
                continue

            # Fill first box predictor
            target[grid_y, grid_x, 0:5] = torch.tensor([x_cell, y_cell, w, h, 1.0])
            # Class one-hot encoding
            target[grid_y, grid_x, 5 * B + label] = 1.0

        return target
    



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


import os
test_dataset = YOLOdataset(
    image_dir="data/test/Images",
    annot_dir="data/test/labels",
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)



def non_max_suppression(boxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes per class.
    boxes: list of [x1, y1, x2, y2, conf, class_idx]
    Returns: list of kept boxes
    """
    if not boxes:
        return []

    # Convert to tensor and validate
    try:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if boxes_tensor.shape[1] != 6:
            warnings.warn(f"Invalid box format: expected 6 elements, got {boxes_tensor.shape[1]}")
            return []
    except Exception as e:
        warnings.warn(f"Failed to convert boxes to tensor: {e}")
        return []

    x1, y1, x2, y2, scores, classes = boxes_tensor[:, 0], boxes_tensor[:, 1], boxes_tensor[:, 2], boxes_tensor[:, 3], boxes_tensor[:, 4], boxes_tensor[:, 5]
    keep_boxes = []

    # Validate box coordinates
    valid_mask = (x2 > x1) & (y2 > y1) & (scores >= 0) & (scores <= 1)
    if not valid_mask.all():
        warnings.warn(f"Invalid boxes detected: {boxes_tensor[~valid_mask].tolist()}")
        boxes_tensor = boxes_tensor[valid_mask]
        if boxes_tensor.shape[0] == 0:
            return []
        x1, y1, x2, y2, scores, classes = boxes_tensor[:, 0], boxes_tensor[:, 1], boxes_tensor[:, 2], boxes_tensor[:, 3], boxes_tensor[:, 4], boxes_tensor[:, 5]

    unique_classes = classes.unique()
    for cls in unique_classes:
        cls_mask = (classes == cls)
        cls_boxes = boxes_tensor[cls_mask]
        if len(cls_boxes) == 0:
            continue
        scores_cls = cls_boxes[:, 4]
        order = scores_cls.sort(descending=True).indices
        cls_boxes = cls_boxes[order]

        while len(cls_boxes) > 0:
            box = cls_boxes[0]  # Keep as tensor
            keep_boxes.append(box.tolist())
            if len(cls_boxes) == 1:
                break
            rest = cls_boxes[1:]
            
            # Compute IoU
            xx1 = torch.max(box[0], rest[:, 0])
            yy1 = torch.max(box[1], rest[:, 1])
            xx2 = torch.min(box[2], rest[:, 2])
            yy2 = torch.min(box[3], rest[:, 3])
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            iou = inter / (area1 + area2 - inter + 1e-6)

            # Keep boxes with IoU below threshold
            cls_boxes = rest[iou < iou_threshold]

    return keep_boxes


def decode_predictions(pred, S=4, B=2, C=20, conf_threshold=0.1, img_shape=(224, 224), nms_iou=0.5):
    """
    Decode model predictions into boxes.
    pred: model output [S, S, C+5*B]
    Returns: list of boxes [x1, y1, x2, y2, conf, class_idx] in pixel coordinates with NMS
    """
    boxes = []
    H, W = img_shape

    # Validate input
    if pred.shape != (S, S, C + 5 * B):
        warnings.warn(f"Invalid prediction shape: expected {(S, S, C + 5 * B)}, got {pred.shape}")
        return []

    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            for b in range(B):
                # Apply sigmoid to confidence and coordinates
                conf = torch.sigmoid(cell[5*b + 4]).item()
                if conf < conf_threshold:
                    continue
                x_cell = torch.sigmoid(cell[5*b]).item()
                y_cell = torch.sigmoid(cell[5*b + 1]).item()
                w = torch.abs(cell[5*b + 2]).item()  # Ensure non-negative
                h = torch.abs(cell[5*b + 3]).item()  # Ensure non-negative

                # Validate coordinates
                if not (0 <= x_cell <= 1 and 0 <= y_cell <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    warnings.warn(f"Invalid box coordinates: [x_cell={x_cell}, y_cell={y_cell}, w={w}, h={h}] at grid ({i}, {j}), box {b}")
                    continue

                # Cell to image scale (normalized [0, 1])
                x_center = (j + x_cell) / S
                y_center = (i + y_cell) / S

                # Convert to pixel coordinates
                x1 = (x_center - w/2) * W
                y1 = (y_center - h/2) * H
                x2 = (x_center + w/2) * W
                y2 = (y_center + h/2) * H

                # Clip boxes
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                # Validate box dimensions
                if x2 <= x1 or y2 <= y1:
                    warnings.warn(f"Invalid box dimensions: [x1={x1}, y1={y1}, x2={x2}, y2={y2}] at grid ({i}, {j}), box {b}")
                    continue

                # Class prediction
                class_probs = torch.softmax(cell[5*B:], dim=0)
                class_idx = torch.argmax(class_probs).item()

                boxes.append([x1, y1, x2, y2, conf, class_idx])

    # Apply NMS
    boxes = non_max_suppression(boxes, iou_threshold=nms_iou)
    if not boxes:
        warnings.warn("No boxes after NMS")
    return boxes


from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_boxes(image, pred_boxes, true_boxes, idx=0, classes=None):
    """
    Plot predicted (red) and ground truth (green) boxes on an image.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))  # CHW to HWC
    for box in pred_boxes:
        if box[0] == idx:
            x1, y1, x2, y2, conf, cls = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if classes:
                ax.text(x1, y1, f"{classes[int(cls)]} ({conf:.2f})", color='r', fontsize=8)
    for box in true_boxes:
        if box[0] == idx:
            x1, y1, x2, y2, conf, cls = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            if classes:
                ax.text(x1, y1, classes[int(cls)], color='g', fontsize=8)
    plt.savefig(f"debug_boxes_batch_{idx}.png")
    plt.close()

model.eval()
pred_boxes, true_boxes = [], []
S = 4
H, W = 224, 224
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

for idx, (x, y) in enumerate(tqdm(test_loader, desc="Evaluating")):
    if x is None or y is None:
        warnings.warn(f"Skipping batch {idx} due to invalid data")
        continue
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    
    # Debug model output
    if idx < 2:  # Debug first two batches
        print(f"Batch {idx} output shape: {out.shape}, min: {out.min().item():.4f}, max: {out.max().item():.4f}")
        print(f"Sample confidence scores: {torch.sigmoid(out[0, :, :, 4::5]).flatten()[:5]}")
    
    batch_pred = decode_predictions(out[0].cpu(), S=4, B=2, C=20, conf_threshold=0.1, img_shape=(W, H), nms_iou=0.5)
    batch_true = cellboxes_to_boxes(y[0], S=4, B=2, C=20, img_size=(W, H))

    # Debug box counts and sample boxes
    if batch_pred:
        print(f"Batch {idx} sample pred box: {batch_pred[0]}")
    if batch_true:
        print(f"Batch {idx} sample GT box: {batch_true[0]}")
    print(f"Batch {idx}: {len(batch_pred)} predictions, {len(batch_true)} ground truth")

    # Visualize first few batches
    if idx < 2 and batch_pred and batch_true:
        plot_boxes(x[0].cpu(), batch_pred, batch_true, idx=0, classes=classes)

    # Append normalized pred boxes
    for box in batch_pred:
        x1, y1, x2, y2, conf, cls = box
        x_center = (x1 + x2) / (2 * W)
        y_center = (y1 + y2) / (2 * H)
        w_box = (x2 - x1) / W
        h_box = (y2 - y1) / H
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w_box <= 1 and 0 <= h_box <= 1):
            warnings.warn(f"Invalid pred box: {box}")
        pred_boxes.append([idx, cls, conf, x_center, y_center, w_box, h_box])

    # Append normalized GT boxes
    for box in batch_true:
        x1, y1, x2, y2, conf, cls = box
        x_center = (x1 + x2) / (2 * W)
        y_center = (y1 + y2) / (2 * H)
        w_box = (x2 - x1) / W
        h_box = (y2 - y1) / H
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w_box <= 1 and 0 <= h_box <= 1):
            warnings.warn(f"Invalid GT box: {box}")
        true_boxes.append([idx, cls, x_center, y_center, w_box, h_box])

# Calculate mAP and class-only accuracy
map_score = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20, size=224)
class_acc = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20, size=224, class_only=True)
print(f"mAP@0.5: {map_score:.4f}")
print(f"Class-only accuracy: {class_acc:.4f}")




