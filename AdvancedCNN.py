import torch 
import torch.nn as nn  # neural network (moduel)


class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Define convolutional blocks with clear naming for readability
        self.conv1 = self._create_conv_block(in_channels=3, out_channels=16)
        self.conv2 = self._create_conv_block(in_channels=16, out_channels=32)
        self.conv3 = self._create_conv_block(in_channels=32, out_channels=64)
        self.conv4 = self._create_conv_block(in_channels=64, out_channels=128)
        self.conv5 = self._create_conv_block(in_channels=128, out_channels=128)

        # Fully-connected layers (assuming image data with flattening)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=2048),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=10),
            nn.Dropout(p=0.5),
            nn.ReLU())
    

    def _create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        

        # Flatten before feeding to fully-connected layers
        x = x.view(x.size(0), -1)  # Adjust based on input dimensions
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    
    import torch

def train(model, train_loader, optimizer, criterion, device, num_epochs):


  model.to(device)  # Move model to the specified device

  for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    running_loss = 0.0  # Initialize running loss for each epoch
    model.train()  # Set model to training mode (important for dropout)

    for i, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)

      # Calculate loss
      loss = criterion(outputs, labels)

      # Backward pass and parameter update
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Update running loss
      running_loss += loss.item()

      if i % 100 == 99:  # Print loss every 100 batches
        print(f"[Batch {i + 1}/{len(train_loader)}] Loss: {running_loss / 100}")
        running_loss = 0.0  # Reset running loss for next batch

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1} Training Loss: {loss.item()}")

  print("Training Complete!")
