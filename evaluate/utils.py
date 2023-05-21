import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.transforms.functional import to_pil_image
from matplotlib.patches import Rectangle
from torchvision.transforms import ToTensor, RandomCrop, Compose, Resize, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.ops as ops
from torchvision.ops import box_iou
import pickle
import cv2


idx_to_label = {
0: 'background',
1:'aeroplane',
2:'bicycle',
3:'bird',
4:'boat',
5:'bottle',
6:'bus',
7:'car',
8:'cat',
9:'chair',
10:'cow',
11:'diningtable',
12:'dog',
13:'horse',
14:'motorbike',
15:'person',
16:'pottedplant',
17:'sheep',
18:'sofa',
19:'train',
20:'tvmonitor'
}

label_to_idx = {y:x for (x,y) in idx_to_label.items()}

def collate_fn(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        target = sample[1]
        
        # Extract bounding boxes from target
        objects = target["annotation"]["object"]
        boxes = []
        labels = []
        for obj in objects:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_to_idx[obj["name"]])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)
        
        # Add boxes to target dictionary
        target = {"boxes": boxes, "labels": labels}
        targets.append(target)
    
    return images, targets



def pick_random_image(data, seed=None):
    if seed != None:
        random.seed(seed)
    idx = random.randint(0, len(data))
    img, target = collate_fn([data[idx]])
    return img, target

def draw_boxes(image, boxes):

    uint = torch.tensor(image * 255, dtype=torch.uint8)
    PIlImg = to_pil_image(uint)
    fig, ax = plt.subplots()
    
    ax.imshow(PIlImg)


    for idx, bbox in enumerate(boxes):

        color = 'r' if idx == 0 else 'g'
        
        (x1, y1, x2, y2) = (bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item())
        bbox_rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox_rect)

    plt.savefig('test.jpg')
    plt.show()



def soft_nms(pred, threshold=0.3, sigma=0.5, method='linear'):
    """
    Apply Soft-NMS to the boxes and scores.
    
    Args:
    - boxes: A tensor of shape (N, 4) containing the boxes in (x1, y1, x2, y2) format.
    - scores: A tensor of shape (N,) containing the confidence scores for each box.
    - threshold: The IoU threshold below which the scores will be reduced.
    - sigma: The weighting parameter for the gaussian function in the 'gaussian' method.
    - method: The method to use for weight calculation: 'linear' or 'gaussian'.
    
    Returns:
    - boxes: A tensor of shape (M, 4) containing the boxes after Soft-NMS.
    - scores: A tensor of shape (M,) containing the scores after Soft-NMS.
    """
    
    boxes, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']


    N = boxes.size(0)
    indexes = torch.arange(N, dtype=torch.float).to(boxes.device)
    
    # Sort the scores in descending order
    _, order = scores.sort(0, descending=True)
    
    boxes = boxes[order, :]
    scores = scores[order]
    labels = labels[order]
    
    for i in range(N):
        # Get the i-th box
        box = boxes[i, :]
        score = scores[i]
        
        # Calculate the IoU between the i-th box and the remaining boxes

        ious = box_iou(box.unsqueeze(0), boxes[i+1:, :])
        ious = torch.squeeze(ious)
        
        # Apply the weighting function
        if method == 'linear':
            weight = torch.ones_like(ious)
            weight[ious > threshold] -= ious[ious > threshold]
        elif method == 'gaussian':
            weight = torch.exp(-(ious ** 2) / sigma)
        else:
            raise ValueError("Unsupported weighting method.")
        
        if len(weight.size()) == 0: break
        # Apply the weighting to the scores
        scores[i+1:] *= weight
        
        # Apply the threshold
        keep = scores[i+1:] >= threshold
        
        # Update the scores and indexes
        scores[i+1:][keep == True] *= weight[keep == True]
        indexes[i+1:][keep == True] = 0
    
    # Remove the suppressed boxes
    keep = (indexes == 0)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    pred[0]['boxes'], pred[0]['scores'], pred[0]['labels'] = boxes, scores, labels
    
    return boxes, scores, labels


def remove_by_index(tensor, indices): #remove an element by index in a tensor

    device = tensor.device
    indices_to_keep = torch.as_tensor([i for i in range(tensor.size(0)) if i not in indices], dtype=torch.int64, device=device)
    return torch.index_select(tensor, 0, indices_to_keep)


def nms(model_output):

    keep = {"boxes": [], "labels": [], "scores": []}
    boxes = model_output[0]["boxes"]
    labels = model_output[0]["labels"]
    scores = model_output[0]["scores"] 

    iou_threshold = 0.5

    while boxes.shape[0] != 0:

        max_score_idx = torch.argmax(scores).item() # get element with max confidence score

        box = boxes[max_score_idx]
        score = scores[max_score_idx]
        label = labels[max_score_idx]
        # print(f"Max score: {score}")

        if score < 0.2:
            break

        selected_box = torch.stack([boxes[max_score_idx]])
 
        boxes = remove_by_index(boxes, [max_score_idx])
        labels = remove_by_index(labels, [max_score_idx])
        scores = remove_by_index(scores, [max_score_idx])
        
        iou_matrix = box_iou(selected_box, boxes)
        # print(iou_matrix)

        inds = torch.where(iou_matrix >= iou_threshold)[1]
        

        keep["boxes"].append(box)
        keep["labels"].append(label)
        keep["scores"].append(score)

        boxes = (remove_by_index(boxes, inds))
        labels = (remove_by_index(labels, inds))
        scores = (remove_by_index(scores, inds))


    keep["boxes"] = torch.stack(keep["boxes"]) if len(keep["boxes"]) > 0 else torch.tensor([])
    keep["labels"] = torch.stack(keep["labels"]) if len(keep["labels"]) > 0 else torch.tensor([])
    keep["scores"] = torch.stack(keep["scores"]) if len(keep["scores"]) > 0 else torch.tensor([])


    return keep


def predict(model, image):
    with torch.no_grad():
        model.eval()
        out = model(torch.stack([image]))
    
    return out


def smooth_l1_loss(bbox_pred, bbox_target, beta=1.0):
    """
    Compute the smooth L1 loss between two bounding boxes.

    Args:
        bbox_pred (torch.Tensor): Predicted bounding box with shape (4,).
        bbox_target (torch.Tensor): Target bounding box with shape (4,).
        beta (float, optional): Smoothing parameter. Default: 1.0.

    Returns:
        torch.Tensor: Smooth L1 loss.
    """
    diff = bbox_pred - bbox_target
    abs_diff = torch.abs(diff)
    smooth_l1 = torch.where(abs_diff < beta, 0.5 * abs_diff ** 2 / beta, abs_diff - 0.5 * beta)
    return smooth_l1.mean()


def translate_bbox(bbox1, translation_vector):
    translate_x = translation_vector[: , 0]
    translate_y = translation_vector[: , 1]

    if (len(bbox1.shape) == 1):
        bbox1 = bbox1.unsqueeze(0)

    bbox1[:, 0] += translate_x
    bbox1[:, 1] += translate_y
    bbox1[:, 2] += translate_x
    bbox1[:, 3] += translate_y

    return bbox1

def scale_bbox(bbox, scale_coef):

    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]   
    width = x_max - x_min
    height = y_max - y_min
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    new_width = width * scale_coef
    new_height = height * scale_coef
    new_x_min = center_x - new_width / 2
    new_y_min = center_y - new_height / 2
    new_x_max = center_x + new_width / 2
    new_y_max = center_y + new_height / 2
    new_bbox = torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max])
    return new_bbox.unsqueeze(0)


def average_cam(cam, bbox):

    if len(bbox.shape) > 1:
        bbox = bbox.squeeze()

    x1 = int(bbox[0].item())
    y1 = int(bbox[1].item())
    x2 = int(bbox[2].item())
    y2 = int(bbox[3].item())

    bbox_cam = cam[y1:y2, x1:x2]

    return np.mean(bbox_cam)

#for each detection assign a ground truth bbox
def assign_bbox(detections, gt_bboxes, iou_threshold=0.5):
    final_detections = []
    final_targets = []

    #sort detections by confidence score
    # detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    for bbox in detections["boxes"]:
        if len(gt_bboxes) == 0:
            break

        iou_matrix = box_iou(bbox.unsqueeze(0), gt_bboxes)

        iou_max, iou_max_idx = torch.max(iou_matrix, dim=1)

        if iou_max > iou_threshold:
            final_detections.append(bbox)
            final_targets.append(gt_bboxes[iou_max_idx])
            gt_bboxes = remove_by_index(gt_bboxes, iou_max_idx)
    
    return torch.stack(final_detections), torch.stack(final_targets)


def interpretation_heatmap(cam, img, pred_bbox, dest_file, contrastive=None):

    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    t_heatmap = heatmap.astype(np.float32)
    t_heatmap = t_heatmap / 255

    # Overlay the heatmap on the input image
    output = cv2.addWeighted(img, 0.5, t_heatmap, 0.5, 0)

    if len(pred_bbox.shape) > 1:
        pred_bbox = pred_bbox.squeeze(0)

    output = cv2.rectangle(output, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 0, 255), 2)
    
    if contrastive != None:
        contrastive = contrastive.squeeze()
        output = cv2.rectangle(output, (int(contrastive[0]), int(contrastive[1])), (int(contrastive[2]), int(contrastive[3])), (0, 255, 0), 2)
    cv2.imwrite(dest_file, output * 255)
    return output


def validate_one_epoch(test_loader, device, model):
    model.train()
    loss = 0
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss += sum(loss for loss in loss_dict.values())
    return loss / len(test_loader)


def compute_feature_maps(img, model):
    
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



def compute_grad_CAM(img, outputs, model):

    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=model.backbone.body.layer4[-1].conv3.weight, #last feature extraction conv layer
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.squeeze()

    weights = torch.mean(grads, dim=1).cpu() # global average pooling shape=(2048,)

    feature_maps = compute_feature_maps(img, model).cpu() # (1, 2048, 12, 16)

    cam = torch.zeros(feature_maps.shape[-2:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[0, i, :, :]

    t_img = img.squeeze(0)
    
    # cam_positive = cam[cam > 0]
    # print("CAM positives: ", cam_positive.shape)

    # cam_negative = cam[cam < 0]
    # print("CAM negatives: ", cam_negative.shape)

    cam = cv2.resize(cam.detach().numpy(), (t_img.shape[2], t_img.shape[1]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam

def smooth_grad_cam(pred_bbox, contrastive_bbox, model, image, num_samples=10, stdev_spread=0.15, contrastive=True):
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
        cam = compute_grad_CAM(pred_bbox, contrastive_bbox, model, noisy_image, contrastive)

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


def guided_backpropagation(img, outputs):    
    

    # Compute the gradients of the output with respect to the input
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=img,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

    outputs.backward(retain_graph=True)
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


def calculate_slope(pred_bbox):
    xmin, ymin, xmax, ymax = pred_bbox.squeeze(0)
    slope = (ymax - ymin) / (xmax - xmin)
    return slope

def calculate_distance(pred_bbox):
    xmin, ymin, xmax, ymax = pred_bbox.squeeze(0)
    distance = torch.sqrt(torch.pow(xmax - xmin, 2) + torch.pow(ymax - ymin, 2))
    return distance