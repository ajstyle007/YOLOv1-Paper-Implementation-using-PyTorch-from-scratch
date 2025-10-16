import torch
import cv2
from torchvision import transforms
from func import YOLOv1, cellboxes_to_boxes, draw_boxes, non_max_suppression
import streamlit as st


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

model = YOLOv1(S=4, B=2, C=20)  # your YOLOv1 architecture
# model.load_state_dict(torch.load("logs_old/last_checkpoint.pth", map_location=device))
checkpoint = torch.load("logs_old/last_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # or 448 if you trained that way
])


# ---- Predict function ----
def predict_image(frame, conf_threshold=0.01, iou_threshold=0.01):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        preds = model(img_tensor)

    boxes = cellboxes_to_boxes(preds.squeeze(0).cpu(), conf_threshold=conf_threshold)
    boxes = non_max_suppression(boxes, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    return boxes

# ---- Draw function ----
def draw_boxes(frame, boxes, class_names):
    h, w, _ = frame.shape
    for (x1, y1, x2, y2, conf, cls_idx) in boxes:
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        label = f"{class_names[int(cls_idx)]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, max(20, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return frame

# ---- Streamlit UI ----
st.title("YOLOv1 Live Detection")

conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.01, key="conf_thresh_slider")
iou_thresh = st.slider("IoU Threshold", 0.0, 1.0, 0.5, key="iou_thresh_slider")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)  # 0 = default camera

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to capture frame")
        break

    boxes = predict_image(frame, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
    frame = draw_boxes(frame, boxes, class_names=[
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "ajay",
        "plant", "sheep", "sofa", "train", "tvmonitor"
    ])

    # Convert BGR to RGB for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

camera.release()