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


seed = 42
model.eval()

img, target = utils.pick_random_image(utils.train_dataset, seed=seed)
img = img[0].to(device)
img.requires_grad = True
out = model(img.unsqueeze(0))
utils.soft_nms(out)
keep = utils.nms(out)

target_bbox = target[0]["boxes"].to(device)

detections, boxes = utils.assign_bbox(keep, target_bbox) # assigns detections and gt pairs
index = random.randint(0, detections.shape[0] - 1) # get random element from detections
pred_bbox = detections[index]
target_bbox = boxes[index]

xmin, ymin, xmax, ymax = pred_bbox

def compute_grad_CAM(coord, model, img):
    coord = coord.unsqueeze(0)

    # Compute the gradients of the output with respect to the last convolutional layer
    grads = torch.autograd.grad(
        outputs=coord,
        inputs=model.backbone.body.layer4[-1].conv3.weight, #last feature extraction conv layer
        grad_outputs=torch.ones_like(coord),
        create_graph=True,
        retain_graph=True
    )[0]

    grads = grads.squeeze()

    weights = torch.mean(grads, dim=1).cpu() # global average pooling shape=(2048,)

    feature_maps = utils.compute_feature_maps(img, model).cpu() # (1, 2048, 12, 16)

    cam = torch.zeros(feature_maps.shape[-2:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[0, i, :, :]

    t_img = img.squeeze(0)
    

    cam = cv2.resize(cam.detach().numpy(), (t_img.shape[2], t_img.shape[1]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam


def guided_backpropagation(img, coord):    
    
    coord = coord.unsqueeze(0)
    # img = img.detach().requires_grad_()

    # Compute the gradients of the output with respect to the input
    grad = torch.autograd.grad(
        outputs=coord,
        inputs=img,
        grad_outputs=torch.ones_like(coord),
        create_graph=True,
        retain_graph=True
    )[0]

    coord.backward(retain_graph=True)
    img_grad = img.grad
    
    img_grad_positive = img_grad[img_grad > 0]
    print("Gradient positives: ", img_grad_positive.shape)

    img_grad_negative = img_grad[img_grad < 0]
    print("Gradient negatives: ", img_grad_negative.shape)


    img_grad[img_grad < 0] = 0 # filter out negative gradients

    # Apply guided backpropagation by masking the negative gradients
    guided_grad = img_grad * (grad > 0).float()
    
    # Normalize the gradients
    guided_grad /= torch.max(torch.abs(guided_grad))
    
    # Convert the guided gradients to a numpy array
    guided_grad = guided_grad.squeeze().detach().cpu().numpy()
    return guided_grad

# cam_xmin = compute_grad_CAM(xmin, model, img.unsqueeze(0))
# cam_ymin = compute_grad_CAM(ymin, model, img.unsqueeze(0))
# cam_xmax = compute_grad_CAM(xmax, model, img.unsqueeze(0))
# cam_ymax = compute_grad_CAM(ymax, model, img.unsqueeze(0))


gbackprop_xmin = guided_backpropagation(img, xmin)
gbackprop_ymin = guided_backpropagation(img, ymin)
gbackprop_xmax = guided_backpropagation(img, xmax)
gbackprop_ymax = guided_backpropagation(img, ymax)

def average_channels(t):
    # Normalize the guided CAM and transform it into shape [:, :] by averaging the channels
    t = t / np.max(t)
    t = np.transpose(t, (1, 2, 0))
    t = np.mean(t, axis=2)
    return t


# average_channels(guided_cam)
gbackprop_xmin = average_channels(gbackprop_xmin)
gbackprop_ymin = average_channels(gbackprop_ymin)
gbackprop_xmax = average_channels(gbackprop_xmax)
gbackprop_ymax = average_channels(gbackprop_ymax)

img = img.permute(1, 2, 0).detach().cpu().numpy()

# output = utils.interpretation_heatmap(cam_xmin, img, pred_bbox, None, f"generated/coords/ResNet-101/cam_xmin{seed}.jpg")
# output = utils.interpretation_heatmap(cam_ymin, img, pred_bbox, None, f"generated/coords/ResNet-101/cam_ymin{seed}.jpg")
# output = utils.interpretation_heatmap(cam_xmax, img, pred_bbox, None, f"generated/coords/ResNet-101/cam_xmax{seed}.jpg")
# output = utils.interpretation_heatmap(cam_ymax, img, pred_bbox, None, f"generated/coords/ResNet-101/cam_ymax{seed}.jpg")

# output = utils.interpretation_heatmap(gbackprop_xmin, img, pred_bbox, None, f"generated/coords/ResNet-101/gbackprop_xmin{seed}.jpg")
# output = utils.interpretation_heatmap(gbackprop_ymin, img, pred_bbox, None, f"generated/coords/ResNet-101/gbackprop_ymin{seed}.jpg")
# output = utils.interpretation_heatmap(gbackprop_xmax, img, pred_bbox, None, f"generated/coords/ResNet-101/gbackprop_xmax{seed}.jpg")
# output = utils.interpretation_heatmap(gbackprop_ymax, img, pred_bbox, None, f"generated/coords/ResNet-101/gbackprop_ymax{seed}.jpg")

# cam = utils.compute_grad_CAM(pred_bbox, contrastive, model, img.unsqueeze(0), contrastive=False)