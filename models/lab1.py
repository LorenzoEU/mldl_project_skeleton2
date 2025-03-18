import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
import numpy as np
# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# Load the dataset
tiny_imagenet_dataset_train = ImageFolder(root='/dataset/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_test = ImageFolder(root='/dataset/tiny-imagenet-200/test', transform=transform)

# Create a DataLoader
dataloader_train = DataLoader(tiny_imagenet_dataset_train, batch_size=128, shuffle=True)
dataloader_test = DataLoader(tiny_imagenet_dataset_test, batch_size=128, shuffle=False)

# Determine the number of classes and samples
num_classes = len(tiny_imagenet_dataset_train.classes)
num_samples = len(tiny_imagenet_dataset_train)

print(f'Number of classes: {num_classes}')
print(f'Number of samples: {num_samples}')

# Function to denormalize image for visualization
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

# Visualize one example for each class for 10 classes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
classes_sampled = []
found_classes = 0

for i, (inputs, classes) in enumerate(dataloader_train):
  for idx, classe in enumerate(classes): # Use idx for index and classe for the class label
    if classe not in classes_sampled:
        classes_sampled.append(classe)
        image = denormalize(inputs[idx]) # Use idx to access the correct image from the batch
        row = found_classes // 5
        col = found_classes % 5
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"Class: {classe.item()}") # Use .item() to get the class value as an integer
        axes[row, col].axis('off')
        found_classes += 1
        if found_classes == 10:
            break
  if found_classes == 10:
    break
plt.show()
