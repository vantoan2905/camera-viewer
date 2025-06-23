import torch
from model.yolo.yolo_net import TrainingYOLONet  # hoặc đúng đường dẫn
from custom_dataset.load_data import LoadDataset
import os
from torchvision import transforms












path_df = r"D:\object_detect_tracking\camera-viewer\data\brain_tumor_copy"

axial_path = os.path.join(path_df, "axial_t1wce_2_class")
train_axial_path_image = os.path.join(axial_path, "images", "train")
train_axial_path_label = os.path.join(axial_path, "labels", "train")
test_axial_path_image = os.path.join(axial_path, "images", "test")
test_axial_path_label = os.path.join(axial_path, "labels", "test")



transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = LoadDataset(train_axial_path_image, train_axial_path_label, transform=transform)
test_dataset = LoadDataset(test_axial_path_image, test_axial_path_label, transform=transform)


# show size of dataset
print("Size of train dataset:", len(train_dataset))
print("Size of test dataset:", len(test_dataset))




# ---------------------------------------------------------------------------
# Initialize the YOLO model
# ---------------------------------------------------------------------------

model = TrainingYOLONet(num_classes=20, num_anchors=3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


print(model)