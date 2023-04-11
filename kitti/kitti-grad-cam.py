import torch
import torchvision
import utils
import cv2
import numpy as np
import random
from kitti_train.KittiDataset import KittiDataset
import torch.nn.functional as F

models_dict = {
    "default_kitti": "models/fasterrcnn_kitti.pth",
    "kitti_reg": "models/fasterrcnn_kitti_reg.pth"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=10)
model.load_state_dict(torch.load(models_dict["default_kitti"], map_location=device))
model.to(device)

model.eval()

dataset = KittiDataset("/data/auto/kitti/object/training")

seed = 5001
img, target = dataset.pick_random_image(seed=seed)
img.requires_grad = True
out = model(img.unsqueeze(0))

# Post-processing
utils.soft_nms(out)
keep = utils.nms(out)

print(f"Prediction (after nms): {keep}")

detections, boxes = utils.assign_bbox(keep, target["boxes"])

random.seed(1)
index = random.randint(0, detections.shape[0]-1) # get random element from detections 
pred_bbox = detections[index]
target_bbox = boxes[index]

copy_bbox = pred_bbox.clone().detach()
contrastive = utils.translate_bbox(copy_bbox, np.array([[50,50]]))



def compute_feature_maps(img):
    
    body = model.backbone.body
    
    img = body.conv1(img)
    img = body.bn1(img)
    img = body.relu(img)
    img = body.maxpool(img)

    img = body.layer1(img)
    img = body.layer2(img)
    img = body.layer3(img)
    img = body.layer4(img)
    
    return img


def interpretation_heatmap(cam, img, pred_bbox, contrastive, dest_file):
    img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_RAINBOW)

    t_heatmap = heatmap.astype(np.float32)
    t_heatmap = t_heatmap / 255

    # Overlay the heatmap on the input image
    output = cv2.addWeighted(img, 0.5, t_heatmap, 0.5, 0)

    if len(pred_bbox.shape) > 1:
        pred_bbox = pred_bbox.squeeze(0)

    output = cv2.rectangle(output, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 0, 255), 2)
    contrastive = contrastive.squeeze()
    output = cv2.rectangle(output, (int(contrastive[0]), int(contrastive[1])), (int(contrastive[2]), int(contrastive[3])), (0, 255, 0), 2)
    cv2.imwrite(dest_file, output * 255)
    return output



def compute_grad_CAM(pred_bbox, contrastive_bbox, model, img):
    loss = utils.smooth_l1_loss(pred_bbox, contrastive_bbox) # since we are dealing with single object, we can just use the first box
    loss = loss.unsqueeze(0)

    # Compute the gradients of the output with respect to the last convolutional layer
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=model.backbone.body.layer4[-1].conv3.weight, #last feature extraction conv layer
        grad_outputs=torch.ones_like(loss),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.squeeze()

    weights = torch.mean(grads, dim=1) # global average pooling shape=(2048,)

    feature_maps = compute_feature_maps(img) # (1, 2048, 12, 16)

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


def guided_backpropagation(img, pred_bbox, contrastive_bbox):    
    
    loss = utils.smooth_l1_loss(pred_bbox, contrastive_bbox)
    loss = loss.unsqueeze(0)

    # Compute the gradients of the output with respect to the input
    grad = torch.autograd.grad(
        outputs=loss,
        inputs=img,
        grad_outputs=torch.ones_like(loss),
        create_graph=True,
        retain_graph=True,
    )[0]

    loss.backward(retain_graph=True)
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
    for i in range(num_samples):
        # Add noise to the input image
        range_value = torch.max(image) - torch.min(image)
        noise = torch.randn_like(image) * range_value * stdev_spread
        noisy_image = image + noise

        # Compute the Grad-CAM for the noisy image
        cam = compute_grad_CAM(pred_bbox, contrastive, model, noisy_image)

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


smooth_grad = smooth_grad_cam(img.unsqueeze(0), num_samples=5, stdev_spread=0.15)
# smooth_grad = smooth_grad.squeeze(0)

cam = compute_grad_CAM(pred_bbox, contrastive, model, img.unsqueeze(0))
guided_grad = guided_backpropagation(img, pred_bbox, contrastive)


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

output = interpretation_heatmap(guided_grad, img, pred_bbox, contrastive, f"generated/guided_grad{seed}.jpg")
output4 = interpretation_heatmap(guided_grad, img, pred_bbox, contrastive, f"generated/guided_cam{seed}.jpg")
output4 = interpretation_heatmap(cam, img, pred_bbox, contrastive, f"generated/cam{seed}.jpg")
output = interpretation_heatmap(smooth_grad, img, pred_bbox, contrastive, f"generated/smooth_grad{seed}.jpg")


# Display the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()