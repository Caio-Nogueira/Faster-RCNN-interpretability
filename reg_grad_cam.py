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

model.eval()

img, target = utils.pick_random_image(utils.train_dataset, seed=42)
img = img[0].to(device)
out = model(img.unsqueeze(0))
utils.soft_nms(out)
keep = utils.nms(out)

detections = utils.assign_bbox(keep["boxes"], target[0]["boxes"])
index = random.randint(0, detections.shape[0]) # get random element from detections 
pred_bbox = detections[index]
target_bbox = target[0]["boxes"][index]

print(target[0]["labels"][index])

copy_bbox = pred_bbox.clone().detach()
contrastive = utils.translate_bbox(copy_bbox, np.array([[50,-50]]))

loss = utils.smooth_l1_loss(pred_bbox, contrastive) # since we are dealing with single object, we can just use the first box
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


feature_maps = compute_feature_maps(img) # (1, 2048, 12, 16)


cam = torch.zeros(feature_maps.shape[-2:], dtype=torch.float32)
for i, w in enumerate(weights):
    cam += w * feature_maps[0, i, :, :]


cam = torch.abs(cam) # all values are negative so we take the absolute value 
cam = cv2.resize(cam.detach().numpy(), (img.shape[2], img.shape[1]))
cam = np.maximum(cam, 0)
cam = cam / cam.max()

print(utils.average_cam(cam, pred_bbox))
print(utils.average_cam(cam, contrastive))

img = np.array(img)

t_img = img.transpose((1, 2, 0))

img = cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR)

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

cv2.imwrite("generated/grad_cam_interpretation3.jpg", output * 255)

# Display the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()