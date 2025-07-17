import os
import torch
from torchvision import transforms
from model.yolo.yolo_net import YOLONet
from custom_dataset.load_data import LoadDataset
from custom_dataset.viz_data import DataVisualizer
# --- Config ---
DATA_ROOT = r"D:\object_detect_tracking\camera-viewer\data\brain_tumor_copy"
NUM_CLASSES = 2
NUM_ANCHORS = 3

# --- Dataset paths ---
axial_path = os.path.join(DATA_ROOT, "axial_t1wce_2_class")
train_img = os.path.join(axial_path, "images", "train")
train_lbl = os.path.join(axial_path, "labels", "train")
test_img = os.path.join(axial_path, "images", "test")
test_lbl = os.path.join(axial_path, "labels", "test")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load datasets ---
train_dataset = LoadDataset(train_img, train_lbl, transform=transform)
test_dataset = LoadDataset(test_img, test_lbl, transform=transform)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")




# --- Model ---
model = YOLONet(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("Model initialized.")
print(model)
# parameters size of model 
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")



# --- Training model with onw image ---

# Get a sample from the training dataset
sample_image, sample_labels = train_dataset[0]

# Add batch dimension
sample_image = sample_image.unsqueeze(0)
sample_labels = sample_labels.unsqueeze(0)

# Forward pass
output = model(sample_image)
# Compute loss
loss = criterion(output, sample_labels)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Training step completed.")

