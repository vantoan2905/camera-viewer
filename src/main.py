import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.yolo.yolo_net import YOLONet
from custom_dataset.load_data import LoadDataset
from custom_dataset.viz_data import DataVisualizer
import os
from torch import nn, optim
import argparse as ap
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

class YOLOModel():
    def __init__(self, mode_model, num_classes, num_anchors, device):
        self.mode_model = mode_model
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.device = device

    def load_model(self):
        if self.mode_model != "yolo":
            raise ValueError(f"Unsupported model type: {self.mode_model}. Only 'yolo' is supported.")
        self.model = YOLONet(num_classes=self.num_classes, num_anchors=self.num_anchors).to(self.device)
        self.model.train()

    def load_dataset(self, data_path, transform):
        train_path_image = os.path.join(data_path, "images/train")
        train_path_label = os.path.join(data_path, "labels/train")
        test_path_image = os.path.join(data_path, "images/test")
        test_path_label = os.path.join(data_path, "labels/test")
        train_dataset = LoadDataset(train_path_image, train_path_label, transform=transform)
        test_dataset = LoadDataset(test_path_image, test_path_label, transform=transform)
        dataset = {
            "train": train_dataset,
            "test": test_dataset
        }
        return dataset

    def visualize_dataset(self, dataset):
        data_viz = DataVisualizer(dataset)
        data_viz.viz_sample_5_images()

def train(model, dataloader, optimizer, criterion, device, epochs, save_dir):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"yolo_epoch_{epoch+1}.pth"))

def main():
    """
    Main function to set up and train a YOLO model based on a configuration file.

    This script parses command-line arguments to obtain the path to a configuration file,
    which contains parameters for training the YOLO model such as image size, number of classes,
    number of anchors, and device to be used. It then loads the configuration, sets up the
    necessary directories, initializes the model, loads the dataset, and begins training.

    The training configuration is printed, and optionally, the dataset is visualized before training.
    
    Raises:
        FileNotFoundError: If the specified configuration file does not exist.

    """

    parser = ap.ArgumentParser(description="YOLO Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} does not exist.")
    config = load_config(args.config)

    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device(config['device'])

    print(f"\nYOLO Training Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    transform = get_transform(config['imgsz'])

    yolo_model = YOLOModel(
        mode_model=config['mode_model'],
        num_classes=config['num_classes'],
        num_anchors=config['num_anchors'],
        device=device
    )
    yolo_model.load_model()

    dataset_dict = yolo_model.load_dataset(data_path=config['data'], transform=transform)
    dataset = dataset_dict['train']
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    if config.get('visualize', False):
        yolo_model.visualize_dataset(dataset)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(yolo_model.model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    train(yolo_model.model, dataloader, optimizer, criterion, device, config['epochs'], config['save_dir'])

if __name__ == "__main__":
    main()
