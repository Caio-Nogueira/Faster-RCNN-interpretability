import torch
import torchvision
import utils
import cv2
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)
model.load_state_dict(torch.load("models/fasterrcnn_10epochs.pth", map_location=device))
model.to(device)


seed = 42
model.eval()

img, target = utils.pick_random_image(utils.train_dataset, seed=seed)
img = img[0].to(device)
img.requires_grad = True
out = model(img.unsqueeze(0))
utils.soft_nms(out)
keep = utils.nms(out)

detections, boxes = utils.assign_bbox(keep, target[0]["boxes"])
index = random.randint(0, detections.shape[0] - 1) # get random element from detections
pred_bbox = detections[index]
target_bbox = boxes[index]


copy_bbox = pred_bbox.clone().detach()
# contrastive = utils.translate_bbox(copy_bbox, np.array([[50,-50]]))
contrastive = utils.scale_bbox(copy_bbox, 1.25)

def mask_image(image, pred_bbox, contrastive):
    pred_bbox = pred_bbox.detach().cpu().numpy().astype(np.int32)
    contrastive = contrastive.detach().cpu().numpy().squeeze(0).astype(np.int32)

    binary_mask = torch.zeros_like(image)
    binary_mask[pred_bbox[1]:pred_bbox[3], pred_bbox[0]:pred_bbox[2]] = 1
    binary_mask[contrastive[1]:contrastive[3], contrastive[0]:contrastive[2]] = 1

    return image * binary_mask


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


def compute_grad_CAM(pred_bbox, contrastive_bbox, img, layer=None, layer_num=None):
    loss = utils.smooth_l1_loss(pred_bbox, contrastive_bbox) # since we are dealing with single object, we can just use the first box
    loss = loss.unsqueeze(0)
    layer.requires_grad = True


    # Compute the gradients of the output with respect to the last convolutional layer
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=layer, #last feature extraction conv layer
        grad_outputs=torch.ones_like(loss),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.squeeze()

    weights = torch.mean(grads, dim=1) # global average pooling shape=(2048,)

    masked_image = mask_image(img.squeeze(0), pred_bbox, contrastive_bbox)

    feature_maps = compute_feature_maps(masked_image, layer_num) # (1, 2048, 12, 16)

    cam = torch.zeros(feature_maps.shape[-2:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[0, i, :, :]

    t_img = img.squeeze(0)
    
    cam_positive = cam[cam > 0]
    print("CAM positives: ", cam_positive.shape)

    cam_negative = cam[cam < 0]
    print("CAM negatives: ", cam_negative.shape)

    cam = cv2.resize(cam.detach().numpy(), (t_img.shape[2], t_img.shape[1]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam

cam2 = compute_grad_CAM(pred_bbox, target_bbox, img.unsqueeze(0), layer=model.backbone.body.layer2[-1].conv3.weight, layer_num=2)
cam3 = compute_grad_CAM(pred_bbox, target_bbox, img.unsqueeze(0), layer=model.backbone.body.layer3[-1].conv3.weight, layer_num=3)
cam4 = compute_grad_CAM(pred_bbox, contrastive, img.unsqueeze(0), layer=model.backbone.body.layer4[-1].conv3.weight, layer_num=4)


img = img.permute(1, 2, 0).detach().cpu().numpy()


output = utils.interpretation_heatmap(cam2, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam2_{seed}.jpg")
output = utils.interpretation_heatmap(cam3, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam3_{seed}.jpg")
output = utils.interpretation_heatmap(cam4, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam4_{seed}.jpg")

final_cam = (cam2 + cam3 + cam4) / 3

pred_bbox = pred_bbox.detach().cpu().numpy().astype(np.int32)
contrastive = contrastive.detach().cpu().numpy().squeeze(0).astype(np.int32)

# mask regions out of both bounding boxes
binary_mask = np.zeros_like(final_cam)
binary_mask[pred_bbox[1]:pred_bbox[3], pred_bbox[0]:pred_bbox[2]] = 1
binary_mask[contrastive[1]:contrastive[3], contrastive[0]:contrastive[2]] = 1

final_cam = final_cam * binary_mask


output = utils.interpretation_heatmap(final_cam, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam_final_{seed}.jpg")