from model.yolo.yolo_net import YOLONet
import torch




if __name__ == "__main__":


    num_classes = 80 
    num_anchors = 3  # Example: 3 anchors, adjust as needed
    
    model = YOLONet(num_classes=num_classes, num_anchors=num_anchors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # Example loss function, adjust as needed
    
    # Dummy data for training
    dummy_input = torch.randn(1, 3, 416, 416)  # Batch size of 1, 3 channels, 416x416 image
    dummy_target = torch.randn(1, num_anchors * (4 + 1 + num_classes), 13, 13)  # Example target tensor
    
    # Training loop
    model.train()
    for epoch in range(10):  # Example: 10 epochs
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
    print("Training completed successfully!")
    
    # Note: This is a simplified example. In practice, you would use a proper dataset, data loaders, and more sophisticated training logic.