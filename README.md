# YOLOv1-Paper-Implementation-using-PyTorch-from-scratch

This project is a from-scratch implementation of the YOLOv1 (You Only Look Once) object detection paper using PyTorch.
I implemented the entire pipeline — architecture, loss function, dataset parsing, and model training — to deeply understand how YOLO works at its core.

<img width="1459" height="769" alt="Screenshot 2025-10-16 161807" src="https://github.com/user-attachments/assets/a161d946-9989-41b7-9ec4-72829a0c0d79" />

<img width="1366" height="733" alt="Screenshot 2025-10-16 162046" src="https://github.com/user-attachments/assets/65b55d9a-0d0d-45c0-b0b2-8eae39d6b8cc" />


## 🚀 What I Built
- 🧩 ResNet-18 Backbone (for feature extraction)
- ⚙️ YOLOv1 Loss Function (custom-built using MSELoss)
- 📦 Custom Dataset Loader for PASCAL VOC 2007 + 2012
- 🕒 15+ Hours of Training with Checkpointing & Logging
- 🧮 Complete Web App using Flask for image detection demo

## 🛠️ Technical Details
- Optimizer: Adam
- Learning Rate Scheduler: StepLR
- Loss Function: MSE with λ_coord = 5, λ_noobj = 0.5
- Dataset: PASCAL VOC 2007 + 2012 (XML annotations parsed)
- Framework: PyTorch
- Web Framework: Flask

## 🧩 YOLOv1 Loss Function

The YOLOv1 loss function combines:

- Localization Loss (for bounding box coordinates)
- Confidence Loss (for objectness score)
- Classification Loss (for class probabilities)

  <img width="980" height="685" alt="loss_func" src="https://github.com/user-attachments/assets/19a88922-e9d9-4e66-9ab8-dc40e379f49c" />

I replicated the official YOLOv1 loss equation and implemented it using torch.nn.MSELoss with custom weighting factors:
``λ_coord = 5.0``
``λ_noobj = 0.5``

This ensures bounding box coordinates are penalized more heavily, while boxes without objects contribute less to the loss.

## 🧱 YOLOv1 Architecture

The YOLOv1 head is built on top of a ResNet-18 backbone pre-trained on ImageNet.
It outputs a grid structure that predicts bounding boxes and class probabilities in a single forward pass, enabling real-time object detection without region proposals.

<img width="1395" height="701" alt="yolo_architechture" src="https://github.com/user-attachments/assets/ffe1ef21-ceb9-442a-bd34-9628a318b096" />

## 🧪 Web App Demo

The Flask web app allows users to upload images and view detection results instantly.
Due to Hugging Face Spaces limitations, the live webcam detection feature is disabled — but you can find a recorded demo video of live detections on my LinkedIn.

## 🧍 Detected Classes

👨 Person 🚗 Car 🐱 Cat 🐶 Dog 🚌 Bus 🚲 Bicycle
✈️ Aeroplane 🚤 Boat 🐄 Cow 🐑 Sheep 🐦 Bird
🏍️ Motorbike 🚂 Train 🛵 Horse 🪑 Chair 🛋️ Sofa
🍽️ Diningtable 📺 TVmonitor 🪴 Pottedplant 🍷 Bottle

## 💬 Final Thoughts

This project became a cornerstone of my AI research journey.
I now have a deep understanding of YOLO’s architecture, loss design, and real-time detection principles — and this project represents my growth from reading research papers to building real implementations.
