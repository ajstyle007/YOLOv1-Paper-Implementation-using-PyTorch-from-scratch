import torch
from torch import nn

class YOLO(nn.Module):
    def __init__(self, B, C):
        super().__init__()

        self.B = B
        self.C = C
        self.dropout = nn.Dropout(0.5)

        
        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels=64, kernel_size=7, stride=2, padding=3) #1
        self.leaky1 = nn.LeakyReLU(0.1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2,stride=2, padding=0) #2

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1) #3
        self.leaky2 = nn.LeakyReLU(0.1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #4

        self.conv_3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0) #5
        self.leaky3 = nn.LeakyReLU(0.1)

        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) #6
        self.leaky4 = nn.LeakyReLU(0.1)

        self.conv_5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0) #7
        self.leaky5 = nn.LeakyReLU(0.1)

        self.conv_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) #8
        self.leaky6 = nn.LeakyReLU(0.1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #9


        # 10–17
        self.conv_7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #10
        self.leaky7 = nn.LeakyReLU(0.1)

        self.conv_8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) #11
        self.leaky8 = nn.LeakyReLU(0.1)

        self.conv_9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #12
        self.leaky9 = nn.LeakyReLU(0.1)

        self.conv_10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) #13
        self.leaky10 = nn.LeakyReLU(0.1)

        self.conv_11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #14
        self.leaky11 = nn.LeakyReLU(0.1)

        self.conv_12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) #15
        self.leaky12 = nn.LeakyReLU(0.1)

        self.conv_13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #16
        self.leaky13 = nn.LeakyReLU(0.1)

        self.conv_14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) #17
        self.leaky14 = nn.LeakyReLU(0.1)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)



        self.conv_15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0) #18
        self.leaky15 = nn.LeakyReLU(0.1)

        self.conv_16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1) #19
        self.leaky16 = nn.LeakyReLU(0.1)

        self.conv_17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0) #20
        self.leaky17 = nn.LeakyReLU(0.1)

        self.conv_18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1) #21
        self.leaky18 = nn.LeakyReLU(0.1)

        self.conv_19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1) #22
        self.leaky19 = nn.LeakyReLU(0.1)

        self.conv_20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1) #23
        self.leaky20 = nn.LeakyReLU(0.1)

        self.conv_21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1) #24
        self.leaky21 = nn.LeakyReLU(0.1)



        self.fc1 = nn.Linear(in_features=7*7*1024, out_features=4096)
        self.leaky22 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(4096, 7*7*(self.B*5 + self.C))

        # B = 2 (bounding boxes per grid cell)
        # C = 20 (for Pascal VOC 20 classes)


    def forward(self, x):

        out1 = self.leaky1(self.conv_1(x))
        out2 = self.max_pool_1(out1)

        out3  = self.leaky2(self.conv_2(out2))
        out4 = self.max_pool_2(out3)

        out4 = self.leaky3(self.conv_3(out4))
        out5 = self.leaky4(self.conv_4(out4))
        out6 = self.leaky5(self.conv_5(out5))
        out7 = self.leaky6(self.conv_6(out6))
        out8 = self.max_pool_3(out7)
        out9 = self.leaky7(self.conv_7(out8))
        out10 = self.leaky8(self.conv_8(out9))
        out11 = self.leaky9(self.conv_9(out10))
        out12 = self.leaky10(self.conv_10(out11))
        out13 = self.leaky11(self.conv_11(out12))
        out14 = self.leaky12(self.conv_12(out13))
        out15 = self.leaky13(self.conv_13(out14))
        out16 = self.leaky14(self.conv_14(out15))
        out17 = self.max_pool_4(out16)
        out18 = self.leaky15(self.conv_15(out17))
        out19 = self.leaky16(self.conv_16(out18))
        out20 = self.leaky17(self.conv_17(out19))
        out21 = self.leaky18(self.conv_18(out20))
        out22 = self.leaky19(self.conv_19(out21))
        out23 = self.leaky20(self.conv_20(out22))
        out24 = self.leaky21(self.conv_21(out23))

        out25 = out24.view(out24.size(0), -1)
        out26 = self.leaky22(self.fc1(out25))
        out27 = self.dropout(out26)
        out28 = self.fc2(out27)


        return out28



B = 2
C = 20
model = YOLO(B, C)
x = torch.randn(1, 3, 448, 448)
out = model(x)
print(out.shape)  # Should be [1, 7*7*(B*5 + C)] >>  [1, 1470]


# loss function of yolo

import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, λ_coord=5, λ_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = λ_coord
        self.lambda_noobj = λ_noobj

    def forward(self, predictions, target):
        # Reshape to [batch_size, S, S, (C + B*5)]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # Yahan:
        # predictions ek bada tensor hoga (model ka output)
        # self.S — grid size hoti hai (YOLO me jaise 7x7)
        # self.B — bounding boxes per grid cell (usually 2)
        # self.C — classes ki sankhya (e.g., 20 for VOC dataset)
        # -1 — automatically dimension adjust kar do (PyTorch/Numpy khud calculate karega)    
        # YOLO jaise models output dete hain ek flat tensor me.
        # Par hume us output ko reshape karna padta hai taki har grid cell ka prediction alag se mil jaye.
        # To agar tumhara model output hai (batch_size, S*S*(C + B*5)),
        # to reshape karke hum usse (batch_size, S, S, C + B*5) me badalte hain
        # taki har grid cell ke andar ke prediction (classes + bounding boxes) ko access kar sakein.


        # --- 1️⃣ Identify cells containing objects ---
        obj_mask = target[..., self.C + 4].unsqueeze(-1)  # 1 if object present
        # YOLO ka target tensor shape hota hai:
        # (S, S, C + 5*B)
        # Agar hum YOLOv1 ki baat karein:
        # S = 7 (grid size)
        # B = 2 (bounding boxes per cell)
        # C = 20 (class categories)
        # To har cell ke liye total output hota hai:
        # C + 5*B = 20 + 10 = 30 features

        # Ek grid cell ke andar values kuch aise hoti hain:
        # [ x, y, w, h, conf, class_1, class_2, ..., class_20,  x2, y2, w2, h2, conf2 ]

        # target[..., self.C + 4] means self.C + 4 = 20 + 4 = 24
        # → ye 24th index hai (0-based indexing), jo represent karta hai pehle bounding box ka confidence score.
        # yaani har cell me agar object present hai, to is position par 1 hoga,
        # aur agar koi object nahi hai, to 0 hoga.

        # .unsqueeze(-1)
        # iska matlab hai tensor ke end me ek extra dimension jod dena,
        # taaki broadcasting me problem na aaye jab hum is mask ko multiply karenge predictions ke saath.
        # Before unsqueeze : (7, 7)	
        # Shape	: mask without extra dim
        # After unsqueeze : (7, 7, 1)

        # 👉 obj_mask ek boolean/tensor mask hai jo batata hai:
        # kis grid cell me object present hai (1)
        # aur kis me nahi (0)




        # --- 2️⃣ Localization loss ---
        box_pred = predictions[..., self.C:self.C + 5]
        box_target = target[..., self.C:self.C + 5]

        # Only for cells containing object
        box_pred_xy = obj_mask * box_pred[..., :2]
        box_target_xy = obj_mask * box_target[..., :2]

        box_pred_wh = obj_mask * torch.sign(box_pred[..., 2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6))
        box_target_wh = obj_mask * torch.sqrt(box_target[..., 2:4])

        coord_loss = self.mse(box_pred_xy, box_target_xy) + self.mse(box_pred_wh, box_target_wh)
        coord_loss = self.lambda_coord * coord_loss

        # --- 3️⃣ Object confidence loss ---
        conf_pred = predictions[..., self.C + 4]
        conf_target = target[..., self.C + 4]

        obj_conf_loss = self.mse(obj_mask.squeeze(-1) * conf_pred, obj_mask.squeeze(-1) * conf_target)

        # --- 4️⃣ No object confidence loss ---
        noobj_mask = 1 - obj_mask
        noobj_conf_loss = self.mse(noobj_mask.squeeze(-1) * conf_pred, noobj_mask.squeeze(-1) * conf_target)
        noobj_conf_loss = self.lambda_noobj * noobj_conf_loss

        # --- 5️⃣ Classification loss ---
        class_pred = predictions[..., :self.C]
        class_target = target[..., :self.C]

        class_loss = self.mse(obj_mask * class_pred, obj_mask * class_target)

        # --- Final total loss ---
        total_loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss
        return total_loss


loss_fn = YoloLoss(S=7, B=2, C=20)
preds = torch.rand((1, 7*7*(20 + 2*5)))  # output from model
target = torch.rand((1, 7, 7, 30))
loss = loss_fn(preds, target)
print("Loss:", loss.item())



