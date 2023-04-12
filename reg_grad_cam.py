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

output = utils.interpretation_heatmap(guided_grad, img, pred_bbox, contrastive, f"generated/guided_grad{seed}.jpg")
output4 = utils.interpretation_heatmap(guided_grad, img, pred_bbox, contrastive, f"generated/guided_cam{seed}.jpg")
output4 = utils.interpretation_heatmap(cam, img, pred_bbox, contrastive, f"generated/cam{seed}.jpg")


# Display the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

