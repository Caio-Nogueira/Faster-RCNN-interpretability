import torch
import torchvision
import utils
import cv2
import numpy as np
import random
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)

backbone = torchvision.models.resnet101(pretrained=True)
resnet_fpn = resnet_fpn_backbone('resnet101', pretrained=True)

model.backbone = resnet_fpn
model.load_state_dict(torch.load("fasterrcnn_voc-101.pth", map_location=device))
model.to(device)


seed = 2049
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
contrastive = utils.translate_bbox(copy_bbox, np.array([[0, 50]])).to(device)
# contrastive = utils.scale_bbox(copy_bbox, 1.25).to(device)



cam = utils.compute_grad_CAM(pred_bbox, contrastive, model, img.unsqueeze(0), contrastive=False)
smooth_grad = utils.smooth_grad_cam(pred_bbox, contrastive, model, img.unsqueeze(0), num_samples=5,
                                    stdev_spread=0.15, 
                                    contrastive=False)

guided_grad = utils.guided_backpropagation(img, pred_bbox, contrastive, contrastive=False)


def average_channels(t):
    # Normalize the guided CAM and transform it into shape [:, :] by averaging the channels
    t = t / np.max(t)
    t = np.transpose(t, (1, 2, 0))
    t = np.mean(t, axis=2)
    return t


# average_channels(guided_cam)
guided_grad = average_channels(guided_grad)

# Multiply the guided gradients with the computed CAM
guided_cam = cam * guided_grad

img = img.permute(1, 2, 0).detach().cpu().numpy()

contrastive = None

output = utils.interpretation_heatmap(guided_grad, img, pred_bbox, contrastive, f"generated/distance/ResNet-101/guided_grad{seed}.jpg")
output4 = utils.interpretation_heatmap(guided_cam, img, pred_bbox, contrastive, f"generated/distance/ResNet-101/guided_cam{seed}.jpg")
output4 = utils.interpretation_heatmap(cam, img, pred_bbox, contrastive, f"generated/distance/ResNet-101/cam{seed}.jpg")
output = utils.interpretation_heatmap(smooth_grad, img, pred_bbox, contrastive, f"generated/distance/ResNet-101/smooth_grad{seed}.jpg")


# Display the output image
# cv2.imshow("Output", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()