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
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor, RandomCrop, Compose, Resize, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.ops as ops
from torchvision.ops import box_iou
import pickle


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


# Define the transformations to apply
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset

train_dataset = torchvision.datasets.VOCDetection(
    root="./data", year="2012", image_set="train",
    download=False, transform=transform)

val_dataset = torchvision.datasets.VOCDetection(
    root="./data", year="2012", image_set="val",
    download=False, transform=transform)


# Define the data loader
train_data_loader = DataLoader(
    train_dataset, batch_size=2, shuffle=False, num_workers=2,
    collate_fn=collate_fn #handles variable length annotations - images with different number of bounding boxes
)

test_data_loader = DataLoader(
    val_dataset, batch_size=2, shuffle=False, num_workers=2,
    collate_fn=collate_fn 
)

# single_object_dataset = []

# for img, annotation in tqdm(val_dataset):
#     if (len(annotation["annotation"]["object"]) == 1): single_object_dataset.append((img,annotation))

#write single_object_dataset to file
# with open('single_object_dataset.pkl', 'rb') as f:
#     single_object_dataset = pickle.load(f)


def pick_random_image(data):
    random.seed(42)
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

    indices_to_keep = torch.as_tensor([i for i in range(tensor.size(0)) if i not in indices], dtype=torch.int64)
    return torch.index_select(tensor, 0, indices_to_keep)


def is_box_contained(box1, box2):
    """
    Check if box1 is completely contained within box2.
    box1 and box2 are tensors of shape (N, 4) containing the (x, y) coordinates of the top-left and bottom-right corners.
    """
    x1, y1, x2, y2 = box1.split(1, dim=1)
    x3, y3, x4, y4 = box2.split(1, dim=1)
    return torch.all((x1 >= x3) & (y1 >= y3) & (x2 <= x4) & (y2 <= y4), dim=1)


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

        if score < 0.2: break 

        selected_box = torch.stack([boxes[max_score_idx]])
        # selected_box_contained = False


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
        # break

    keep["boxes"] = torch.stack(keep["boxes"])
    keep["labels"] = torch.stack(keep["labels"])
    keep["scores"] = torch.stack(keep["scores"])


    return keep


def predict(model, image):
    # targets = [{k: v.to(device) for k, v in t.items()} for t in target]
    # boxes = target[0]["boxes"]
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