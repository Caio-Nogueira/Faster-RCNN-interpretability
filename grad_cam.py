import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary 
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
# summary(model, (3, 224, 224))
# Set the model to evaluation mode
model.eval()

# Define the preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the image and apply the preprocessing transformations
img = Image.open('example_image.jpg')
img_tensor = transform(img)
# print(img_tensor.shape)

img_tensor = img_tensor.unsqueeze(0)
# print(img_tensor.shape)

# Get the class index corresponding to the predicted class
output = model(img_tensor)
predicted_class_idx = torch.argmax(output).item()

# print(output.shape)
# print(output[:, 188])
# print(model.layer4[-1].conv3.weight.shape)

def convert_shape(image_tensor):
    cnn = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
        nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
    )
    return cnn(image_tensor)


img_tensor = convert_shape(image_tensor=img_tensor)
# print(img_tensor.shape)



# Compute the gradients of the output with respect to the last convolutional layer
grads = torch.autograd.grad(
    outputs=output[:, predicted_class_idx],
    inputs=model.layer4[-1].conv3.weight,
    grad_outputs=torch.ones_like(output[:, predicted_class_idx]),
    create_graph=True,
    retain_graph=True
)[0]


# Compute the weights by taking the global average pooling of the gradients
weights = torch.mean(grads, axis=(2, 3))
print(weights.shape)

# Get the feature maps of the last convolutional layer
feature_maps = model.layer4(model.layer3(model.layer2(model.layer1(img_tensor))))

# Compute the class activation map by computing the weighted sum of the feature maps
cam = torch.zeros(feature_maps.shape[2:], dtype=torch.float32)
for i, w in enumerate(weights[0]):
    cam += w * feature_maps[0, i, :, :]


# Upsample the class activation map to match the size of the input image
cam = cv2.resize(cam.detach().numpy(), (img.size[0], img.size[1]))
cam = np.maximum(cam, 0)
cam = cam / cam.max()

# Convert the input image to a numpy array
img = np.array(img)

# t_img = img.transpose((1, 2, 0))

# Normalize the image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = img.astype(np.float32) / 255.0

# Create a heatmap by applying a color map to the class activation map
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)


# t_heatmap = heatmap.reshape(img.shape)
heatmap = heatmap.astype(np.float32)

# print(heatmap.shape, img.shape)

# Overlay the heatmap on the input image
output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
# output = cv2.rectangle(output, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), (0, 0, 0), 2)

# Display the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
