from transformers import Dinov2Model, Dinov2Config
# from transformers import Dinov2ImageProcessor
from transformers import AutoImageProcessor, Dinov2ForImageClassification
from PIL import Image
import torch
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Load the DINOv2 model (choose the appropriate size: small, base, large, etc.)
model_name = "facebook/dinov2-base"
model = Dinov2Model.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

input_size = 224  # Input size for the model

# Load and preprocess the image
image_path = "rgb_bn.png"
image = Image.open(image_path).convert("RGB")
# print(image.size)

# Use the DINOv2 image processor
processor = AutoImageProcessor.from_pretrained(model_name)
inputs = processor(images=image, return_tensors="pt")

# Get the pixel values (preprocessed image)
pixel_values = inputs["pixel_values"]

# Extract features
with torch.no_grad():
    outputs = model(pixel_values)
    feature_maps = outputs.last_hidden_state  # Shape: [batch_size, num_patches, feature_dim]

# Add a simple segmentation head
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Reshape feature maps to 2D
batch_size, num_patches, feature_dim = feature_maps.shape

# Remove the class token (the first token)
patch_features = feature_maps[:, 1:, :]  # Shape: [batch_size, num_patches, feature_dim]
print(patch_features.shape)

# Calculate the patch grid size
patch_size = 14  # DINOv2 uses 14x14 patches
grid_size = input_size // patch_size  # Number of patches along each dimension

feature_maps_2d = patch_features.reshape(batch_size, grid_size, grid_size, feature_dim).permute(0, 3, 1, 2)

# Define the segmentation head
num_classes = 2  # Example: binary segmentation (background and foreground)
segmentation_head = SegmentationHead(in_channels=feature_dim, out_channels=num_classes)

# Generate the segmentation mask
with torch.no_grad():
    segmentation_output = segmentation_head(feature_maps_2d)
    segmentation_mask = torch.argmax(segmentation_output, dim=1).squeeze().cpu().numpy()  # Shape: [height, width]

# Resize the segmentation mask to match the original image size
original_size = image.size
segmentation_mask = cv2.resize(segmentation_mask, original_size, interpolation=cv2.INTER_NEAREST)

# Extract boundaries using Canny edge detection
boundaries = cv2.Canny((segmentation_mask * 255).astype(np.uint8), 100, 200)

# Overlay boundaries on the original image
overlay = np.array(image).copy()
overlay[boundaries != 0] = [255, 0, 0]  # Highlight boundaries in red

# Visualize the results
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Segmentation mask
plt.subplot(1, 3, 2)
plt.imshow(segmentation_mask, cmap="jet")
plt.title("Segmentation Mask")
plt.axis("off")

# Boundaries overlaid on the original image
plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Boundaries Overlay")
plt.axis("off")

plt.show()