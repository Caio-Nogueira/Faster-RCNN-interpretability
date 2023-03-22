import torch
import torchvision
import utils
import cv2
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)
model.load_state_dict(torch.load("fasterrcnn_10epochs.pth", map_location=device))


img = None
target_bbox = None

with open('example_image_tensor.pkl', 'rb') as f1:
    img = torch.load(f1, map_location=device)


with open('example_image_target.pkl', 'rb') as f2:
    target_bbox = torch.load(f2, map_location=device)

model.to(device)
model.eval()


pred_bbox = model(img.unsqueeze(0))[0]['boxes'][0] # bbox with the highest score

loss = utils.smooth_l1_loss(pred_bbox, target_bbox) # since we are dealing with single object, we can just use the first box
loss = loss.unsqueeze(0)


# Compute the gradients of the output with respect to the last convolutional layer
grads = torch.autograd.grad(
    outputs=loss,
    # inputs=model.backbone.fpn.layer_blocks[-1][0].weight, #last feature extraction conv layer
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

# print(fasterrcnn_reshape_transform(feature_maps).shape)

feature_maps = compute_feature_maps(img) # (1, 2048, 12, 16)


# cam = torch.sum(weights * feature_maps, dim=(1,2), keepdim=True)

cam = torch.zeros(feature_maps.shape[-2:], dtype=torch.float32)
for i, w in enumerate(weights):
    cam += w * feature_maps[0, i, :, :]


cam = torch.abs(cam) # all values are negative (idk why) so we take the absolute value 
cam = cv2.resize(cam.detach().numpy(), (img.shape[2], img.shape[1]))
cam = np.maximum(cam, 0)
cam = cam / cam.max()

img = np.array(img)

t_img = img.transpose((1, 2, 0))

img = cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

t_heatmap = heatmap.astype(np.float32)

# Overlay the heatmap on the input image
output = cv2.addWeighted(img, 0.5, t_heatmap, 0.5, 0)
output = cv2.rectangle(output, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), (0, 0, 0), 2)

# cv2.imwrite("grad_cam_interpretation.jpg", output)
# Display the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
