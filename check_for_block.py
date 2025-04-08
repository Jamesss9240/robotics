import math
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def take_picture():
    #cv2.namedWindow("Camera")
    cap = cv2.VideoCapture(1)  
    
        
    ret, frame = cap.read()
    cap.release() 
    
    if not ret:
        print("Failed to capture image")
        return None
        
    cv2.imshow("Camera", frame)
    cv2.waitKey(1)  
    #save to images folder
    images_folder = "images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    image_file = os.path.join(images_folder, "captured_image.jpg")
    cv2.imwrite(image_file, frame)
    print(f"Image saved as {image_file}")
    return frame, image_file

    

def rotateToAudio(Lmic, Rmic, Fmic, Bmic, heading):
    currenthead = heading
    #find 2 mics with largest sound volume
    mics = [Lmic, Rmic, Fmic, Bmic]
    mics.sort(reverse = True)
    mics = mics[:2]
    if Bmic and Rmic in mics:
        
        currenthead = currenthead - 90
    elif Bmic and Lmic in mics:
        
        currenthead = currenthead + 90
 
    side1 =  10**(mics[0]/10)
    side2 =  10**(mics[1]/10)
    angleRad = math.degrees(math.acos(side2/side1))
    angle = angleRad * 180 / math.pi
    if Lmic == mics[0] or Lmic == mics[1]:
        currenthead = currenthead + angle
       
    else:
        currenthead = currenthead - angle
    return currenthead
    
def pictest():
    # Camera and object parameters
    camera_fov = 90
    KNOWN_BALL_DIAMETER = 100
  
    #cv2.namedWindow("Block Detection")
    ball_found = False
    angle_to_turn = None
    
    # Ask user for input option
    option = int(input("Choose option (1: Take picture, 2: Load image): "))
    
    # Set up device and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './cnn_hsv_hybrid.pth'  # Path to the hybrid model
    
    try:
        # Load the hybrid model with HSV parameters
        # Add weights_only=False to fix the loading error
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract model state dict and HSV parameters
        model_state_dict = model_data['model_state_dict']
        hsv_params = model_data['hsv_params']
        color_mapping = model_data.get('color_mapping', {})
        
      
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
                # Attention mechanism
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
                
        # Create and load model
        model = ConvNet().to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print("Hybrid model loaded successfully!")
    except Exception as e:
        print(f"Error loading hybrid model: {e}")
        print("Falling back to traditional HSV detection only")
        model = None
        hsv_params = {
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
    
    # Image processing transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Define hybrid prediction function
    def hybrid_prediction(image, hsv_params, model=None):
        # Convert OpenCV image to PIL for model input if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Create HSV image for color detection
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get HSV-based color detection
        # Create masks for red (which wraps around the hue spectrum)
        mask_red1 = cv2.inRange(hsv_img, hsv_params['red']['lower1'], hsv_params['red']['upper1'])
        mask_red2 = cv2.inRange(hsv_img, hsv_params['red']['lower2'], hsv_params['red']['upper2'])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Create mask for blue
        mask_blue = cv2.inRange(hsv_img, hsv_params['blue']['lower'], hsv_params['blue']['upper'])
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.erode(mask_red, kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, kernel, iterations=2)
        mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
        
        # Find contours
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contours
        largest_red = max(contours_red, key=cv2.contourArea, default=None) if contours_red else None
        largest_blue = max(contours_blue, key=cv2.contourArea, default=None) if contours_blue else None
        
        hsv_color = 'none'
        hsv_confidence = 0.0
        hsv_contour = None
        
        # Determine color from HSV
        red_area = cv2.contourArea(largest_red) if largest_red is not None else 0
        blue_area = cv2.contourArea(largest_blue) if largest_blue is not None else 0
        
        if red_area > blue_area and red_area > 100:
            hsv_color = 'red'
            hsv_confidence = red_area / (image.shape[0] * image.shape[1])
            hsv_contour = largest_red
        elif blue_area > red_area and blue_area > 100:
            hsv_color = 'blue'
            hsv_confidence = blue_area / (image.shape[0] * image.shape[1])
            hsv_contour = largest_blue
        
        # If model is available, get CNN prediction
        cnn_color = 'none'
        cnn_confidence = 0.0
        
        if model is not None:
            try:
                # Convert to tensor and get prediction
                image_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    _, predicted = torch.max(outputs, 1)
                    predicted_idx = predicted.item()
                    
                    # Map prediction to color (depends on your model's output mapping)
                    if predicted_idx == 0:
                        cnn_color = 'blue'  # Adjust based on your model
                    elif predicted_idx == 2:  
                        cnn_color = 'red'   # Adjust based on your model
                    else:
                        cnn_color = 'none'
                        
                    cnn_confidence = probabilities[predicted_idx].item()
            except Exception as e:
                print(f"Error in CNN prediction: {e}")
        
        # Combine HSV and CNN predictions
        final_color = 'none'
        final_contour = None
        
        if hsv_confidence > 0.05:  # If HSV detection is confident
            final_color = hsv_color
            final_contour = hsv_contour
        elif cnn_confidence > 0.7:  # If CNN is confident but HSV isn't
            final_color = cnn_color
            # If CNN detected a color but HSV didn't find a contour for that color
            if final_color == 'red' and largest_red is not None:
                final_contour = largest_red
            elif final_color == 'blue' and largest_blue is not None:
                final_contour = largest_blue
        
        return final_color, final_contour, cnn_confidence
    
    # Process images for ball detection
    for i in range(10):
        if option == 1:
            # Take a picture with camera
            try:
                frame, image_file = take_picture()
                if frame is None:
                    print("Failed to capture image, trying again...")
                    continue
            except Exception as e:
                print(f"Error capturing image: {e}")
                continue
        else:
            # Option 2: Automatically select an image from the images folder
            if i == 0:  # Only process one image
                try:
                    # List all images in the images folder
                    images_folder = "images"
                    if not os.path.exists(images_folder):
                        print(f"Images folder '{images_folder}' not found. Creating it...")
                        os.makedirs(images_folder)
                        print("Please add images to the folder and run again.")
                        break
                        
                    image_files = [f for f in os.listdir(images_folder) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    
                    if not image_files:
                        print(f"No image files found in {images_folder}. Please add some images.")
                        break
                        
                    print(f"Found {len(image_files)} images. Processing first image...")
                    image_file = image_files[0]
                    image_path = os.path.join(images_folder, image_file)
                    
                    print(f"Loading image: {image_path}")
                    frame = cv2.imread(image_path)
                    
                    if frame is None:
                        print("Failed to load image. Exiting...")
                        break
                        
                except Exception as e:
                    print(f"Error loading image: {e}")
                    break
            else:
                # We've already processed an image in the first iteration
                break
        
        if frame is None:
            print("No valid frame to process, skipping...")
            continue
            
        # Get image dimensions
        height, width, channels = frame.shape
        image_center_x = width // 2
        
        # Use hybrid model for detection
        ball_color, ball_contour, cnn_confidence = hybrid_prediction(frame, hsv_params, model)
        
        # Process detection results
        ball_found = ball_contour is not None and ball_color != 'none'
        
        # Copy frame for visualization
        display_frame = frame.copy()
        
        if ball_found:
            # Set highlight color based on detected color
            highlight_color = (0, 255, 0) if ball_color == "red" else (0, 165, 255)  # Green for red, orange for blue
            
            # Calculate center of ball
            M = cv2.moments(ball_contour)
            if M["m00"] != 0:
                ball_center_x = int(M["m10"] / M["m00"])
                ball_center_y = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(ball_contour)
                ball_center_x = x + w // 2
                ball_center_y = y + h // 2
            
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(ball_contour)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), highlight_color, 2)
            
            # Draw center point
            cv2.circle(display_frame, (ball_center_x, ball_center_y), 5, (255, 255, 255), -1)
            
            # Calculate angle to turn
            focal_length = (width / 2) / math.tan(math.radians(camera_fov / 2))
            pixels_from_center = ball_center_x - image_center_x
            angle_to_turn = (pixels_from_center / (width/2)) * (camera_fov/2)
            turn_direction = 'right' if angle_to_turn > 0 else 'left'
            
            # Calculate distance to ball
            distance = (KNOWN_BALL_DIAMETER * focal_length) / w
            
            # Format angle for display
            if turn_direction == 'left':
                angle_to_turn = -abs(angle_to_turn)
            else:
                angle_to_turn = abs(angle_to_turn)
            
            # Add text to image
            text = f"{ball_color} ball: Turn {int(round(angle_to_turn))}° {turn_direction} confidence: = {cnn_confidence}"
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, highlight_color, 2)
            cv2.putText(display_frame, f"Distance: {distance:.0f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, highlight_color, 2)
            
            print(f"{ball_color.capitalize()} ball found in {image_file}! Width: {w}px, Turn {abs(angle_to_turn):.1f} degrees {turn_direction}, Distance: {distance:.0f}mm")
            
            # Save processed image
            output_dir = "processed_images"
            os.makedirs(output_dir, exist_ok=True)
            
            # Format angle for filename
            angle_str = f"{int(round(angle_to_turn))}"
            
            detection_path = os.path.join(output_dir, f"turn_{angle_str}deg_{ball_color}_{image_file}")
            cv2.imwrite(detection_path, display_frame)
            print(f"{ball_color} ball detection saved as {detection_path}")
        else:
            cv2.putText(display_frame, "No ball detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"No ball detected in {image_file}")
        
        # Show image
        cv2.imshow("Block Detection", display_frame)
        cv2.waitKey(1000)  # Wait for 1 second
    
    cv2.destroyAllWindows()
    return ball_found, angle_to_turn if ball_found else None

# Call the function
pictest()
