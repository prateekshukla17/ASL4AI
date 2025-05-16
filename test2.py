import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model2 import SimpleASLCNN

# Paths and parameters
DATA_DIR = "/Users/anushkatiwari/Desktop/sem stuff/sem 6/ai app/asl-gcn/data/asl_alphabet_test"
BATCH_SIZE = 32
NUM_CLASSES = 29
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataset and DataLoader
test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = SimpleASLCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("../models/cnn_model.pth", map_location=DEVICE))
model.eval()

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")