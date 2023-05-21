import torch
import torchvision
import utils
import cv2
import numpy as np
import random
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)

# backbone = torchvision.models.resnet101(pretrained=True)
# resnet_fpn = resnet_fpn_backbone('resnet101', pretrained=True)

# model.backbone = resnet_fpn
model.load_state_dict(torch.load("fasterrcnn_voc-50.pth", map_location=device))
model.to(device)


seed = 42
model.eval()

img, target = utils.pick_random_image(utils.train_dataset, seed=seed)
img = img[0].to(device)
img.requires_grad = True
out = model(img.unsqueeze(0))
utils.soft_nms(out)
keep = utils.nms(out)

target_bbox = target[0]["boxes"].to(device)

detections, boxes = utils.assign_bbox(keep, target_bbox)
index = random.randint(0, detections.shape[0] - 1) # get random element from detections
pred_bbox = detections[index]
target_bbox = boxes[index]


copy_bbox = pred_bbox.clone().detach().cpu()
# contrastive = utils.translate_bbox(copy_bbox, np.array([[0, 50]])).to(device)
contrastive = utils.scale_bbox(copy_bbox, 1.25).to(device)


cam = utils.compute_grad_CAM(pred_bbox, contrastive, model, img.unsqueeze(0))


def apply_threshold(array, threshold):
    thresholded_array = np.zeros_like(array)
    thresholded_array[array >= threshold] = 1
    return thresholded_array

thr_cam = apply_threshold(cam, 0.5)

def mask_bbox(contrastive, cam):
    xmin, ymin, xmax, ymax = contrastive.squeeze(0)

    for i in range(cam.shape[0]):
        for j in range(cam.shape[1]):
            if i < ymin or i > ymax or j < xmin or j > xmax: 
                cam[i][j] = 0

    return cam

def contrastive_eval(cam, contrastive):
    thr_cam = apply_threshold(cam, 0.5)
    thr_cam = mask_bbox(contrastive, thr_cam)
    return np.sum(thr_cam) / utils.calculate_area(contrastive)


print(f"Contrastive eval: {contrastive_eval(cam, contrastive)}")
