import utils
import torch
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.detection import mean_ap
from tqdm import tqdm
import torch.nn as nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.ops import box_iou

train_dataset = utils.train_dataset
val_dataset = utils.val_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)
model.load_state_dict(torch.load("fasterrcnn_voc-50.pth", map_location=device))
model.to(device)

    #*# Train
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# n_epochs = 30
# delta = 3
# counter = 0
# best_loss = float('inf')


# epoch_losses = []
# epoch_vals = []
# model.train()
# for epoch in range(n_epochs):
#     total_loss = 0
    
#     for images, targets in tqdm(train_loader):
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         optimizer.zero_grad()
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         total_loss += losses.item()
#         losses.backward()
#         optimizer.step()
#         all_losses = {x: y.item() for (x,y) in loss_dict.items()}

    
#         #* Validation
#     val_loss = utils.validate_one_epoch(test_loader, device, model)
#     if val_loss < best_loss:
#         best_loss = val_loss
#         counter = 0
    
#     else:
#         counter += 1
#         if counter > delta:
#             print("Early stopping")
#             break

        
#     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, val_loss: {val_loss}")
#     epoch_losses.append(total_loss)
#     epoch_vals.append(val_loss)

# print(epoch_losses)



#*# Test
model.eval()
# all_predictions = []
# all_targets = []

# mAP = mean_ap.MeanAveragePrecision()

total_objects = 0


with torch.no_grad():
    for images, targets in tqdm(test_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        out = model(images)
        utils.soft_nms(out)
        keep = utils.nms(out)
        keep = [{k: v.cpu() for k, v in keep.items()}]

        # mAP.update(keep, targets)

        # all_predictions.extend(keep)
        # all_targets.extend(targets)

    #* Calculate MSE loss
        
        total_objects = 0
        for i, target in enumerate(targets):
            total_loss = 0
            batch_objects = 0
            for j, box in enumerate(target['boxes']):
                if keep[i]['boxes'].size(0) == 0: continue # No objects detected

                iou_matrix = box_iou(keep[i]['boxes'], box.unsqueeze(0))
                iou_max, iou_max_idx = torch.max(iou_matrix, dim=0)
                
                pred_box = keep[i]['boxes'][iou_max_idx]
                batch_objects += 1
                total_loss += utils.smooth_l1_loss(pred_box, box)
            
            if batch_objects != 0: 
                total_L1 = total_loss / batch_objects
            total_objects += batch_objects

L1_loss = total_L1 / total_objects
print(f"L1 loss: {L1_loss}")

# mAP_value = mAP.compute()
# mAP_value2 = mAP(all_predictions, all_targets)
# print(f"mAP: {mAP_value}")
# print(f"mAP2: {mAP_value2}")



# torch.save(model.state_dict(), '/home/up201806218/Desktop/fasterrcnn_voc-50.pth')
