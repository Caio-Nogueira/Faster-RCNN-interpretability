import torch
import utils
import torchvision
import lime
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)
model = utils.load_pretrained_model(model, "models/fasterrcnn_10epochs.pth", device)
model.to(device)


seed = 2042
model.eval()

# pick random image and select random bbox
img, target = utils.pick_random_image(utils.train_dataset, seed=seed)
img = torch.stack(img).to(device)



def predict_xmin(img): #receives np.array and predicts single coordinate
    out = model(img)
    utils.soft_nms(out)
    keep = utils.nms(out)

    box = keep["boxes"][0]
    xmin = box[0].item() / 255
    # print(xmin)

    return xmin
    


predict_xmin(img)