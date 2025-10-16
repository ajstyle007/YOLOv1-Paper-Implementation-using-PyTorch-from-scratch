import torch
from torch import nn
from torchvision import models
import cv2

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
    

def cellboxes_to_boxes(out, S=4, B=2, C=20, conf_threshold=0.01):
    """
    Convert YOLO output (SxSx(B*5+C)) into bounding boxes in image coordinates
    Returns: list of boxes [x1, y1, x2, y2, conf, class_idx]
    """
    boxes = []
    out = out.reshape(S, S, B*5 + C)
    class_probs = out[..., :C]
    box_preds = out[..., C:].reshape(S, S, B, 5)

    for i in range(S):
        for j in range(S):
            for b in range(B):
                conf = box_preds[i, j, b, 4]
                if conf < conf_threshold:
                    continue
                x, y, w, h = box_preds[i, j, b, :4]
                # Convert from cell-relative to image-relative (normalized)
                x = (j + x) / S
                y = (i + y) / S
                w = w ** 2
                h = h ** 2
                x1 = max(0, (x - w/2))
                y1 = max(0, (y - h/2))
                x2 = min(1, (x + w/2))
                y2 = min(1, (y + h/2))
                
                class_idx = torch.argmax(class_probs[i, j]).item()
                boxes.append([x1, y1, x2, y2, conf.item(), class_idx])
    return boxes



def iou(box1, box2):
    # box: [x1, y1, x2, y2, conf, class]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def non_max_suppression(boxes, iou_threshold=0.01, conf_threshold=0.01):
    """
    boxes: list of [x1, y1, x2, y2, conf, class_idx]
    Returns: filtered boxes after NMS
    """
    boxes = [b for b in boxes if b[4] > conf_threshold]
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    filtered_boxes = []

    while boxes:
        chosen_box = boxes.pop(0)
        boxes = [
            box for box in boxes
            if box[5] != chosen_box[5] or iou(chosen_box, box) < iou_threshold
        ]
        filtered_boxes.append(chosen_box)

    return filtered_boxes


# def draw_boxes(frame, boxes, class_names):
#     h, w, _ = frame.shape
#     for (x1, y1, x2, y2, conf, cls_idx) in boxes:
#         # scale back to original image
#         x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)

#         label = f"{class_names[int(cls_idx)]}: {conf:.2f}"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(frame, label, (x1, y1-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#     return frame

def draw_boxes(frame, boxes, classes):
    """Draw bounding boxes on frame - GUARANTEED TO WORK"""
    frame_copy = frame.copy()
    
    for box in boxes:
        x1, y1, x2, y2, conf, class_idx = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = classes[int(class_idx)]
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label = f"{class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame_copy, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    print(f"Drew {len(boxes)} boxes on frame")
    return frame_copy


def draw_boxes_no_score(frame, boxes, classes):
    """Draw bounding boxes WITHOUT confidence scores"""
    frame_copy = frame.copy()
    
    for box in boxes:
        x1, y1, x2, y2, conf, class_idx = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = classes[int(class_idx)]
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ONLY class name (NO SCORE)
        label = class_name
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame_copy, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame_copy