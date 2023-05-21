import torch
import torchvision
import utils
import cv2
import numpy as np
import random
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)

# backbone = torchvision.models.resnet101(pretrained=True)
# resnet_fpn = resnet_fpn_backbone('resnet101', pretrained=True)

# model.backbone = resnet_fpn
model.load_state_dict(torch.load("fasterrcnn_voc-50.pth", map_location=device))
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


distance = utils.calculate_distance(pred_bbox)
slope = utils.calculate_slope(pred_bbox)
h_dist = utils.calculate_horizontal_distance(pred_bbox)


def compute_feature_maps(img, layer_num=None):
    
    body = model.backbone.body
    
    img = body.conv1(img)
    img = body.bn1(img)
    img = body.relu(img)
    img = body.maxpool(img)

    layers = [body.layer1, body.layer2, body.layer3, body.layer4]

    for l in range(layer_num):
        img = layers[l](img)
    
    return img


def compute_grad_CAM(coord, img, layer=None, layer_num=None):
    coord = coord.unsqueeze(0).to(device)

    layer.requires_grad = True


    # Compute the gradients of the output with respect to the last convolutional layer
    grads = torch.autograd.grad(
        outputs=coord,
        inputs=layer, #last feature extraction conv layer
        grad_outputs=torch.ones_like(coord),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.squeeze()

    weights = torch.mean(grads, dim=1).cpu() # global average pooling 

    # masked_image = mask_image(img.squeeze(0), pred_bbox, contrastive_bbox)

    feature_maps = compute_feature_maps(img, layer_num).cpu() # compute feature maps with respect to a certain layer

    cam = torch.zeros(feature_maps.shape[-2:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[0, i, :, :]

    t_img = img.squeeze(0)
    

    cam = cv2.resize(cam.detach().numpy(), (t_img.shape[2], t_img.shape[1]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam

cam2 = compute_grad_CAM(h_dist, img.unsqueeze(0), layer=model.backbone.body.layer2[-1].conv3.weight, layer_num=2)
cam3 = compute_grad_CAM(h_dist, img.unsqueeze(0), layer=model.backbone.body.layer3[-1].conv3.weight, layer_num=3)
cam4 = compute_grad_CAM(h_dist, img.unsqueeze(0), layer=model.backbone.body.layer4[-1].conv3.weight, layer_num=4)


img = img.permute(1, 2, 0).detach().cpu().numpy()


output = utils.interpretation_heatmap(cam2, img, pred_bbox, None, f"generated/alt_gradCAM/cam2_{seed}.jpg")
output = utils.interpretation_heatmap(cam3, img, pred_bbox, None, f"generated/alt_gradCAM/cam3_{seed}.jpg")
output = utils.interpretation_heatmap(cam4, img, pred_bbox, None, f"generated/alt_gradCAM/cam4_{seed}.jpg")



