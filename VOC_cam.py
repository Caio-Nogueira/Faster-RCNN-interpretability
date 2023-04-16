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


seed = 2
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
    # loss.requires_grad = True


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

    # masked_image = mask_image(img.squeeze(0), pred_bbox, contrastive_bbox)

    feature_maps = compute_feature_maps(img, layer_num) # (1, 2048, 12, 16)

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


def smooth_grad_cam(image, num_samples=10, stdev_spread=0.15):
    """
    Computes the SmoothGrad-CAM for a given input image and model.

    Args:
        image (torch.Tensor): the input image, with shape (3, H, W).
        model (torch.nn.Module): the neural network model.
        layer_name (str): the name of the layer to compute Grad-CAM for.
        num_samples (int): the number of noisy samples to generate.
        stdev_spread (float): the standard deviation of the Gaussian noise,
            as a fraction of the dynamic range of the input image.

    Returns:
        torch.Tensor: the SmoothGrad-CAM heatmap, with shape (1, H', W').
    """

    # Compute the Grad-CAM for each noisy image
    cam_sum = None

    layer = model.backbone.body.layer4[-1].conv3.weight
    layer_num = 4
    for i in range(num_samples):
        # Add noise to the input image
        range_value = torch.max(image) - torch.min(image)
        noise = torch.randn_like(image) * range_value * stdev_spread
        noisy_image = image + noise

        # Compute the Grad-CAM for the noisy image
        cam = compute_grad_CAM(pred_bbox, contrastive, noisy_image, layer, layer_num)

        # Add the CAM to the sum
        if cam_sum is None:
            cam_sum = cam
        else:
            cam_sum += cam

    # Average the CAM maps
    cam_avg = cam_sum / num_samples

    # Apply ReLU activation
    # cam_avg = F.relu(cam_avg)

    # Normalize the CAM
    cam_avg = cam_avg - cam_avg.min()
    cam_avg = cam_avg / cam_avg.max()

    # Resize the CAM to the input image size
    # cam_avg = F.interpolate(cam_avg, size=image.shape[-2:], mode="bilinear", align_corners=False)


    return cam_avg

smoothGrad = smooth_grad_cam(img.unsqueeze(0), num_samples=10, stdev_spread=0.15)


img = img.permute(1, 2, 0).detach().cpu().numpy()


output = utils.interpretation_heatmap(cam2, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam2_{seed}.jpg")
output = utils.interpretation_heatmap(cam3, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam3_{seed}.jpg")
output = utils.interpretation_heatmap(cam4, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam4_{seed}.jpg")

final_cam = (cam2 + cam3 + cam4) / 3


# pred_bbox = pred_bbox.detach().cpu().numpy().astype(np.int32)
# contrastive = contrastive.detach().cpu().numpy().squeeze(0).astype(np.int32)

# # mask regions out of both bounding boxes
# binary_mask = np.zeros_like(final_cam)
# binary_mask[pred_bbox[1]:pred_bbox[3], pred_bbox[0]:pred_bbox[2]] = 1
# binary_mask[contrastive[1]:contrastive[3], contrastive[0]:contrastive[2]] = 1

# final_cam = final_cam * binary_mask



output = utils.interpretation_heatmap(smoothGrad, img, pred_bbox, contrastive, f"generated/global_gradCAM/smoothgrad{seed}.jpg")
output = utils.interpretation_heatmap(final_cam, img, pred_bbox, contrastive, f"generated/global_gradCAM/cam_final_{seed}.jpg")