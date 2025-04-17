import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define HSV color ranges for ball detection (save these values for later use)
HSV_PARAMS = {
    'red': {
        'lower1': np.array([0, 100, 100]),
        'upper1': np.array([10, 255, 255]),
        'lower2': np.array([160, 100, 100]),
        'upper2': np.array([180, 255, 255])
    },
    'blue': {
        'lower': np.array([100, 100, 100]), 
        'upper': np.array([130, 255, 255])
    }
}

num_epochs = 30
batch_size = 4
learning_rate = 0.0013

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.ImageFolder(root='./images',
                                               transform=transform_train)

val_dataset = torchvision.datasets.ImageFolder(root='./val_images',
                                              transform=transform_val)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False)

classes = ('red_block', 'blue_block', 'fake')

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Training class mapping: {train_dataset.class_to_idx}")
print(f"Validation class mapping: {val_dataset.class_to_idx}")

# Class to store our class mapping and color information
class ColorClassInfo:
    def __init__(self, class_to_idx):
        self.class_to_idx = class_to_idx
        # Create reverse mapping
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        # Create color mapping
        self.idx_to_color = {}
        for class_name, idx in class_to_idx.items():
            if 'red' in class_name.lower():
                self.idx_to_color[idx] = 'red'
            elif 'blue' in class_name.lower():
                self.idx_to_color[idx] = 'blue'
            else:
                self.idx_to_color[idx] = 'none'

# Create the color class info
color_class_info = ColorClassInfo(train_dataset.class_to_idx)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(8*8*32, 4)
        
    def forward(self, x):
    # Add batch dimension if input is 3D
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        out = self.layer1(x)
        out = self.layer2(out)
        
        # Apply attention (optional)
        # attention_mask = self.attention(out)
        # out = out * attention_mask
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# HSV color detection function
def detect_color_using_hsv(image_tensor):
    """
    Uses HSV color filtering to detect red or blue objects in an image tensor
    Returns: detected_color list, confidence_score list
    """
    batch_size = image_tensor.size(0)
    detected_colors = []
    confidences = []
    
    # Denormalize and convert to numpy images
    for i in range(batch_size):
        img = image_tensor[i].cpu().numpy().transpose(1, 2, 0)
        img = ((img * 0.5) + 0.5) * 255.0  # Denormalize
        img = img.astype(np.uint8)
        
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Create masks for red (which wraps around the hue spectrum)
        mask_red1 = cv2.inRange(hsv_img, HSV_PARAMS['red']['lower1'], HSV_PARAMS['red']['upper1'])
        mask_red2 = cv2.inRange(hsv_img, HSV_PARAMS['red']['lower2'], HSV_PARAMS['red']['upper2'])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Create mask for blue
        mask_blue = cv2.inRange(hsv_img, HSV_PARAMS['blue']['lower'], HSV_PARAMS['blue']['upper'])
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel for 32x32 images
        mask_red = cv2.erode(mask_red, kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, kernel, iterations=1)
        mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)
        
        # Count pixels for each color
        red_pixels = cv2.countNonZero(mask_red)
        blue_pixels = cv2.countNonZero(mask_blue)
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        
        # Calculate color confidence scores
        red_confidence = red_pixels / total_pixels
        blue_confidence = blue_pixels / total_pixels
        
        # Determine detected color
        if red_confidence > 0.03 and red_confidence > blue_confidence:
            detected_colors.append('red')
            confidences.append(red_confidence)
        elif blue_confidence > 0.03 and blue_confidence > red_confidence:
            detected_colors.append('blue')
            confidences.append(blue_confidence)
        else:
            detected_colors.append('none')
            confidences.append(0.0)
    
    return detected_colors, confidences

# Create custom loss function that incorporates HSV information
class HybridLoss(nn.Module):
    def __init__(self, color_class_info, hsv_weight=0.3):
        super(HybridLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.color_class_info = color_class_info
        self.hsv_weight = hsv_weight
        
    def forward(self, outputs, labels, images, hsv_colors, hsv_confidences):
        # Standard classification loss
        ce_loss = self.cross_entropy(outputs, labels)
        
        # HSV guidance loss
        hsv_loss = 0.0
        batch_size = outputs.size(0)
        
        for i in range(batch_size):
            label_idx = labels[i].item()
            expected_color = self.color_class_info.idx_to_color[label_idx]
            detected_color = hsv_colors[i]
            
            # If HSV detection matches ground truth label's color, reduce loss
            if expected_color == detected_color and detected_color != 'none':
                # Encourage the correct prediction by reducing loss proportional to HSV confidence
                hsv_loss -= hsv_confidences[i] * F.log_softmax(outputs[i], dim=0)[label_idx]
            
        hsv_loss = hsv_loss / batch_size
        
        # Combine losses
        total_loss = (1 - self.hsv_weight) * ce_loss + self.hsv_weight * hsv_loss
        return total_loss, ce_loss, hsv_loss

model = ConvNet().to(device)
criterion = HybridLoss(color_class_info, hsv_weight=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

# Training loop with HSV integration
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_hsv_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Run HSV color detection on batch
        hsv_colors, hsv_confidences = detect_color_using_hsv(images)
        
        # Forward pass
        outputs = model(images)
        
                # Calculate hybrid loss
        loss, ce_loss, hsv_loss = criterion(outputs, labels, images, hsv_colors, hsv_confidences)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_hsv_loss += hsv_loss  # Remove .item() - hsv_loss is already a float
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], '
                  f'Total Loss: {loss:.4f}, CE Loss: {ce_loss:.4f}, HSV Loss: {hsv_loss:.4f}')
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        
        # Track HSV agreement with model predictions
        hsv_model_agreement = 0
        hsv_detected_count = 0
        
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get HSV detections 
            hsv_colors, hsv_confidences = detect_color_using_hsv(images)
            
            # Get CNN predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Check HSV and model agreement
            for i in range(labels.size(0)):
                pred_idx = predicted[i].item()
                pred_color = color_class_info.idx_to_color[pred_idx]
                hsv_color = hsv_colors[i]
                
                if hsv_color != 'none':
                    hsv_detected_count += 1
                    if pred_color == hsv_color:
                        hsv_model_agreement += 1
        
        hsv_agreement_rate = 0
        if hsv_detected_count > 0:
            hsv_agreement_rate = 100.0 * hsv_model_agreement / hsv_detected_count
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * val_correct / val_total:.2f}%')
        print(f'HSV-Model Agreement: {hsv_agreement_rate:.2f}% on {hsv_detected_count} HSV detections')

print('Finished Training')

# Save both the model and HSV parameters
model_data = {
    'model_state_dict': model.state_dict(),
    'hsv_params': HSV_PARAMS,
    'class_mapping': color_class_info.idx_to_class,
    'color_mapping': color_class_info.idx_to_color
}

PATH = './cnn_hsv_hybrid.pth'
torch.save(model_data, PATH)

# Create a hybrid prediction function
def hybrid_prediction(model, image, device, hsv_params):
    """
    Uses both CNN and HSV-based detection for more robust color prediction
    Returns: final_prediction, confidence, position_data
    """
    model.eval()
    
    # Convert image for CNN
    if not isinstance(image, torch.Tensor):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
    else:
        image_tensor = image.to(device)
    
    # Get CNN prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        cnn_prediction = predicted.item()
        cnn_confidence = probabilities[cnn_prediction].item()
    
    # Convert image for HSV detection
    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = ((img * 0.5) + 0.5) * 255.0  # Denormalize
        img = img.astype(np.uint8)
    else:
        img = np.array(image)
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Create masks using saved HSV params
    mask_red1 = cv2.inRange(hsv_img, hsv_params['red']['lower1'], hsv_params['red']['upper1'])
    mask_red2 = cv2.inRange(hsv_img, hsv_params['red']['lower2'], hsv_params['red']['upper2'])
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv_img, hsv_params['blue']['lower'], hsv_params['blue']['upper'])
    
    # Find contours in the masks
    red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour for each color
    largest_red = max(red_contours, key=cv2.contourArea, default=None) if red_contours else None
    largest_blue = max(blue_contours, key=cv2.contourArea, default=None) if blue_contours else None
    
    # Determine which color has the larger contour
    red_area = cv2.contourArea(largest_red) if largest_red is not None else 0
    blue_area = cv2.contourArea(largest_blue) if largest_blue is not None else 0
    
    hsv_color = 'none'
    hsv_confidence = 0.0
    position_data = None
    
    # Get total pixels for confidence calculation
    total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
    
    if red_area > blue_area and red_area > 100:
        hsv_color = 'red'
        hsv_confidence = red_area / total_pixels
        M = cv2.moments(largest_red)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(largest_red)
            position_data = {'center': (cx, cy), 'bbox': (x, y, w, h)}
    
    elif blue_area > red_area and blue_area > 100:
        hsv_color = 'blue'
        hsv_confidence = blue_area / total_pixels
        M = cv2.moments(largest_blue)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(largest_blue)
            position_data = {'center': (cx, cy), 'bbox': (x, y, w, h)}
    
    # Combine CNN and HSV predictions
    if hsv_confidence > 0.05:
        final_color = hsv_color
        final_confidence = hsv_confidence
    else:
        final_color = color_class_info.idx_to_color[cnn_prediction]
        final_confidence = cnn_confidence
    
    return final_color, final_confidence, position_data

# Test the hybrid model
print("\nTesting hybrid model on validation samples:")
model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]
    
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(images.size(0)):
            img = images[i]
            label = labels[i].item()
            
            # Use hybrid prediction
            color, conf, pos = hybrid_prediction(model, img, device, HSV_PARAMS)
            
            # Map color back to class index
            pred_idx = -1
            for idx, clr in color_class_info.idx_to_color.items():
                if clr == color:
                    pred_idx = idx
                    break
            
            if pred_idx == -1:  # If no matching color found, use CNN
                outputs = model(img.unsqueeze(0))
                _, predicted = torch.max(outputs, 1)
                pred_idx = predicted.item()
            
            n_samples += 1
            n_correct += (pred_idx == label)
            
            if label < len(n_class_samples):
                n_class_samples[label] += 1
                if label == pred_idx:
                    n_class_correct[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the hybrid model on validation images: {acc:.2f} %')
    
    for i in range(len(classes)):
        if n_class_samples[i] > 0:
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc:.2f} %')