import torch
import torchvision
import cv2
import numpy as np
from kitti.KittiDataset import KittiDataset
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=10)
model.load_state_dict(torch.load("models/fasterrcnn_kitti.pth", map_location=device))
model.to(device)

model.eval()

dataset = KittiDataset("/data/auto/kitti/object/training")

seed = 2049
img, target = dataset.pick_random_image(seed=seed)
img.requires_grad = True
out = model(img.unsqueeze(0))


layers = [model.backbone.body.layer1, model.backbone.body.layer2, model.backbone.body.layer3, model.backbone.body.layer4]

def compute_feature_maps(img, num_layers):

    body = model.backbone.body
    img = body.conv1(img)
    img = body.bn1(img)
    img = body.relu(img)
    img = body.maxpool(img)

    for i in range(num_layers):
        img = layers[i](img)
    
    return img


def transform_feature_map(tensor):
    tensor = tensor.detach().numpy().squeeze()
    tensor = tensor.transpose(1,2,0)
    tensor = tensor.mean(axis=2)
    tensor = cv2.resize(tensor, (img.shape[2], img.shape[1]))
    tensor = torch.as_tensor(tensor, dtype=torch.float32)

    print(f"Tensor shape: {tensor.unsqueeze(0).shape}")
    
    return tensor.unsqueeze(0)


feature_maps = []

for i in range(4):
    tensor = compute_feature_maps(img.unsqueeze(0), i)
    tensor = transform_feature_map(tensor)
    
    feature_maps.append(tensor)


grid = torchvision.utils.make_grid(feature_maps, nrow=4, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    #save image as png
    plt.savefig("feature_maps.png")


show(grid)