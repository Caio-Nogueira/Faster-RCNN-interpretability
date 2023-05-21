import random
import torch
from torchvision import datasets, transforms
import torchvision
import utils
import numpy as np
from tqdm import tqdm
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


OUT_TYPE = "slope"

def generate_random_integers(k, seed, limit):
    random.seed(seed)  # Set the seed for reproducibility
    random_integers = random.sample(range(1, limit), k)  # Generate k random integers without limiting the range
    return random_integers

def verify_hit(img, model):
    #generate cam heatmap and verify if highest activated pixel is inside the bbox
    out = model(img.unsqueeze(0))
    
    #* Post-processing
    utils.soft_nms(out)
    keep = utils.nms(out)
    
    if len(keep["boxes"]) == 0: return False

    bbox = keep["boxes"][0] #consider first bbox

    if OUT_TYPE == "slope":
        outputs = utils.calculate_slope(bbox)
    
    elif OUT_TYPE == "distance":
        outputs = utils.calculate_distance(bbox)    

    #* Generate CAM heatmap
    cam = cam = utils.compute_grad_CAM(img, outputs, model)

    #* Verify if highest activated pixel is inside the bbox
    cam_max = np.argmax(cam)
    cam_max = np.unravel_index(cam_max, cam.shape)

    #* Verify if highest activated pixel is inside the bbox
    xmin, ymin, xmax, ymax = bbox

    if cam_max[0] >= ymin and cam_max[0] <= ymax and cam_max[1] >= xmin and cam_max[1] <= xmax:
        return True
    
    return False   



if __name__ == '__main__':

    dataset = datasets.Kitti(root='./data', train=False, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    samples = generate_random_integers(1000, 42, len(dataset))

    #* Loading the model
    model_rn101 = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=10)
    backbone = torchvision.models.resnet101()
    resnet_fpn = resnet_fpn_backbone('resnet101', weights=None)
    model_rn101.backbone = resnet_fpn
    model_rn101.load_state_dict(torch.load("models/fasterrcnn_resnet101_kitti_reg.pth"))
    model_rn101.to(device)
    model_rn101.eval()
    
    hits = 0
    print(f"Evaluating ResNet-101 {OUT_TYPE}...")

    for sample in tqdm(samples):
        idx = sample
        img = dataset[idx][0]
        img = transform(img).to(device)
        if verify_hit(img, model_rn101): hits += 1

    print(f"F-RCNN ResNet-101 reg PG metric: {hits/len(samples)}")
