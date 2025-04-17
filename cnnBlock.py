#MODEL TRAINING
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
#training parameter setup
num_epochs = 30
batch_size = 4
learning_rate = 0.0013


transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColourJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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


class ColourClassInfo:
    def __init__(self, class_to_idx):
        self.class_to_idx = class_to_idx
        #reverse mapping
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        #colour mapping
        self.idx_to_colour = {}
        for class_name, idx in class_to_idx.items():
            if 'red' in class_name.lower():
                self.idx_to_colour[idx] = 'red'
            elif 'blue' in class_name.lower():
                self.idx_to_colour[idx] = 'blue'
            else:
                self.idx_to_colour[idx] = 'none'

# Create the colour class info
colour_class_info = ColourClassInfo(train_dataset.class_to_idx)

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
            x = x.unsqueeze(0)  
            
        out = self.layer1(x)
        out = self.layer2(out)
        
  
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def detect_colour_using_hsv(image_tensor):
   
    batch_size = image_tensor.size(0)
    detected_colours = []
    confidences = []
    
  
    for i in range(batch_size):
        img = image_tensor[i].cpu().numpy().transpose(1, 2, 0)
        img = ((img * 0.5) + 0.5) * 255.0  
        img = img.astype(np.uint8)
        
        #convert to bgr for  opencv
        img_bgr = cv2.cvtColour(img, cv2.COLOR_RGB2BGR)
        hsv_img = cv2.cvtColour(img_bgr, cv2.COLOR_BGR2HSV)
        
        #create colour masks
        mask_red1 = cv2.inRange(hsv_img, HSV_PARAMS['red']['lower1'], HSV_PARAMS['red']['upper1'])
        mask_red2 = cv2.inRange(hsv_img, HSV_PARAMS['red']['lower2'], HSV_PARAMS['red']['upper2'])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        
        mask_blue = cv2.inRange(hsv_img, HSV_PARAMS['blue']['lower'], HSV_PARAMS['blue']['upper'])
        
        #reduce noise in image for clarity
        kernel = np.ones((2, 2), np.uint8) 
        mask_red = cv2.erode(mask_red, kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, kernel, iterations=1)
        mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)
        
        #count pixels
        red_pixels = cv2.countNonZero(mask_red)
        blue_pixels = cv2.countNonZero(mask_blue)
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        
        #confidence calculations
        red_confidence = red_pixels / total_pixels
        blue_confidence = blue_pixels / total_pixels
        
       
        if red_confidence > 0.03 and red_confidence > blue_confidence:
            detected_colours.append('red')
            confidences.append(red_confidence)
        elif blue_confidence > 0.03 and blue_confidence > red_confidence:
            detected_colours.append('blue')
            confidences.append(blue_confidence)
        else:
            detected_colours.append('none')
            confidences.append(0.0)
    
    return detected_colours, confidences


class HybridLoss(nn.Module):
    def __init__(self, colour_class_info, hsv_weight=0.3):
        super(HybridLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.colour_class_info = colour_class_info
        self.hsv_weight = hsv_weight
        
    def forward(self, outputs, labels, images, hsv_colours, hsv_confidences):       
        ce_loss = self.cross_entropy(outputs, labels)
        hsv_loss = 0.0
        batch_size = outputs.size(0)
        
        for i in range(batch_size):
            label_idx = labels[i].item()
            expected_colour = self.colour_class_info.idx_to_colour[label_idx]
            detected_colour = hsv_colours[i]
            if expected_colour == detected_colour and detected_colour != 'none':
                hsv_loss -= hsv_confidences[i] * F.log_softmax(outputs[i], dim=0)[label_idx]            
        hsv_loss = hsv_loss / batch_size
        #figure out loss value
        total_loss = (1 - self.hsv_weight) * ce_loss + self.hsv_weight * hsv_loss
        return total_loss, ce_loss, hsv_loss

model = ConvNet().to(device)
criterion = HybridLoss(colour_class_info, hsv_weight=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

#training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_hsv_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        #hsv colour detection
        hsv_colours, hsv_confidences = detect_colour_using_hsv(images)
        
        #forward pass
        outputs = model(images)
        
                #calculate with hybritd approach (hsv+model)
        loss, ce_loss, hsv_loss = criterion(outputs, labels, images, hsv_colours, hsv_confidences)
        
        #backward pass+optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_hsv_loss += hsv_loss 
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], '
                  f'Total Loss: {loss:.4f}, CE Loss: {ce_loss:.4f}, HSV Loss: {hsv_loss:.4f}')
    
    #validate model to test performance
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        
        
        hsv_model_agreement = 0
        hsv_detected_count = 0
        
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
    
            hsv_colours, hsv_confidences = detect_colour_using_hsv(images)
            
            #get prediciton from cnn
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            #check for agreement between cnn+hsv values
            for i in range(labels.size(0)):
                pred_idx = predicted[i].item()
                pred_colour = colour_class_info.idx_to_colour[pred_idx]
                hsv_colour = hsv_colours[i]
                
                if hsv_colour != 'none':
                    hsv_detected_count += 1
                    if pred_colour == hsv_colour:
                        hsv_model_agreement += 1
        
        hsv_agreement_rate = 0
        if hsv_detected_count > 0:
            hsv_agreement_rate = 100.0 * hsv_model_agreement / hsv_detected_count
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * val_correct / val_total:.2f}%')
        print(f'HSV-Model Agreement: {hsv_agreement_rate:.2f}% on {hsv_detected_count} HSV detections')

print('Finished Training!')

#save model
model_data = {
    'model_state_dict': model.state_dict(),
    'hsv_params': HSV_PARAMS,
    'class_mapping': colour_class_info.idx_to_class,
    'colour_mapping': colour_class_info.idx_to_colour
}

PATH = './cnn_hsv_hybrid.pth'
torch.save(model_data, PATH)


def hybrid_prediction(model, image, device, hsv_params):
    model.eval()
    if not isinstance(image, torch.Tensor):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
    else:
        image_tensor = image.to(device)
    
    #cnn predicition
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        cnn_prediction = predicted.item()
        cnn_confidence = probabilities[cnn_prediction].item()
    
    #convert for better hsv detection
    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = ((img * 0.5) + 0.5) * 255.0 
        img = img.astype(np.uint8)
    else:
        img = np.array(image)
    
    img_bgr = cv2.cvtColour(img, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColour(img_bgr, cv2.COLOR_BGR2HSV)
    
    #colour masks
    mask_red1 = cv2.inRange(hsv_img, hsv_params['red']['lower1'], hsv_params['red']['upper1'])
    mask_red2 = cv2.inRange(hsv_img, hsv_params['red']['lower2'], hsv_params['red']['upper2'])
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv_img, hsv_params['blue']['lower'], hsv_params['blue']['upper'])
    
    #find shape conours within masks, and determine which may be a block
    red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_red = max(red_contours, key=cv2.contourArea, default=None) if red_contours else None
    largest_blue = max(blue_contours, key=cv2.contourArea, default=None) if blue_contours else None
    red_area = cv2.contourArea(largest_red) if largest_red is not None else 0
    blue_area = cv2.contourArea(largest_blue) if largest_blue is not None else 0
    hsv_colour = 'none'
    hsv_confidence = 0.0
    position_data = None
    total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
    #closer blocks will generally have a better level of confidence, and be prioritised
    if red_area > blue_area and red_area > 100:
        hsv_colour = 'red'
        hsv_confidence = red_area / total_pixels
        M = cv2.moments(largest_red)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(largest_red)
            position_data = {'center': (cx, cy), 'bbox': (x, y, w, h)}
    
    elif blue_area > red_area and blue_area > 100:
        hsv_colour = 'blue'
        hsv_confidence = blue_area / total_pixels
        M = cv2.moments(largest_blue)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(largest_blue)
            position_data = {'center': (cx, cy), 'bbox': (x, y, w, h)}
    
#combine cnn and hsv prediction, to allow for an accurate result despite lower amounts of training data
    if hsv_confidence > 0.05:
        final_colour = hsv_colour
        final_confidence = hsv_confidence
    else:
        final_colour = colour_class_info.idx_to_colour[cnn_prediction]
        final_confidence = cnn_confidence
    
    return final_colour, final_confidence, position_data

#model testing
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
            
           
            colour, conf, pos = hybrid_prediction(model, img, device, HSV_PARAMS)
            
            #map to class index
            pred_idx = -1
            for idx, clr in colour_class_info.idx_to_colour.items():
                if clr == colour:
                    pred_idx = idx
                    break
            
            if pred_idx == -1:  
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
