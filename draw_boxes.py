import torch
import torchvision
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt

seed = 1042 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)
model.load_state_dict(torch.load("models/fasterrcnn_10epochs.pth", map_location=device))
model.to(device)
model.eval()

img, target = utils.pick_random_image(utils.train_dataset, seed=seed)
img = img[0].to(device)
img.requires_grad = True
out = model(img.unsqueeze(0))
utils.soft_nms(out)
keep = utils.nms(out)


def draw_boxes(image, boxes):
    image = np.transpose(image, (1, 2, 0))

    """
    Draws bounding boxes on the input image and returns the resulting image.
    
    Args:
    - image: numpy.ndarray - the input image
    - boxes: list of tuples - each tuple contains the (x, y, width, height) of a bounding box
    
    Returns:
    - numpy.ndarray - the resulting image with bounding boxes drawn on it
    """
    # Convert the image to a format that OpenCV can handle
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image *= 255
    
    # Draw each bounding box on the image
    for box in boxes:
        box = box.to(torch.int64)
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1.item(), y1.item()), (x2.item(), y2.item()), (0, 0, 1), 2)
    
    # Convert the image back to RGB format and return it
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img = img.detach().cpu().numpy()

image = draw_boxes(img, keep["boxes"])

image = image*255
image = image.astype(np.uint8)

plt.imshow(image)
plt.savefig("test.png")