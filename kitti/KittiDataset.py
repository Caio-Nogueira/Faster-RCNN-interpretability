import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random

#Kitti label_to_idx
label_to_idx = {
    "background": 0,
    "Car": 1,
    "Van": 2,
    "Truck": 3,
    "Pedestrian": 4,
    "Person_sitting": 5,
    "Cyclist": 6,
    "Tram": 7,
    "Misc": 8,
    "DontCare": 9
}


class KittiDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = os.listdir(os.path.join(data_dir, 'image_2'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, 'image_2', self.image_files[idx])
        label_path = os.path.join(self.data_dir, 'label_2', self.image_files[idx].replace('.png', '.txt'))

        image = Image.open(image_path)
        label = self.parse_label(label_path)

        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        image = transform(image)

        return image, label

    def parse_label(self, label_path):
        # parse label file
        # return a list of bounding boxes
        # each bounding box is a list of [xmin, ymin, xmax, ymax, class_id]~
        lines = []
        label = {"boxes": [], "labels": []}
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.split()
            bbox = ([float(x) for x in line[4:8]])
            label["boxes"].append(torch.Tensor(bbox))
            label["labels"].append(label_to_idx[line[0]])

        label["boxes"] = torch.stack(label["boxes"])
        label["labels"] = torch.Tensor(label["labels"]).to(torch.int64)
        return label
    
    def pick_random_image(self, seed=None):
        
        if seed is not None:
            random.seed(seed)

        idx = random.randint(0, len(self.image_files))
        image, label = self[idx]
        return image, label

# Create a data loader
dataset = KittiDataset('/data/auto/kitti/object/training')
