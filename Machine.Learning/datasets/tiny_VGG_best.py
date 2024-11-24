from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from become_1with_data import train_dir , test_dir
# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Simple transformations for testing
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Create DataLoaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, optimizer, and scheduler
from model import Tiny_VGG_v2
import torch
from torch import nn
from helper import train
from torch.optim.lr_scheduler import StepLR

model = Tiny_VGG_v2(input_channels=3, hidden_layers=16, output_classes=len(train_dataset.classes))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# StepLR scheduler (decays LR every 5 epochs)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Use your train function
results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=20,
    scheduler=scheduler
)