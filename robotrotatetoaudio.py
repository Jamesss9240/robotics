import math
import cv2
import os
import numpy as np


def rotateToAudio(Lmic, Rmic, Fmic, Bmic, heading):
    currenthead = heading
    #find 2 mics with largest sound volume
    mics = [Lmic, Rmic, Fmic, Bmic]
    mics.sort(reverse = True)
    mics = mics[:2]
    if Bmic and Rmic in mics:
        #rotate 90 clockwise
        currenthead = currenthead - 90
    elif Bmic and Lmic in mics:
        #rotate 90 counterclockwise
        currenthead = currenthead + 90
    #find the angle and use trig to make robot face the user
    #convert decimal sound to linear linear
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
    # Define color range for ball detection (adjust these HSV values for your ball color)
    # Example for an orange ball
    lower_color = (5, 100, 100)  # HSV lower bounds
    upper_color = (15, 255, 255) # HSV upper bounds
    
    # Camera parameters
    camera_fov = 90  # Horizontal field of view in degrees
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    ball_found = False
    
    for i in range(10):
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        
        original_frame = frame.copy()
        height, width, channels = frame.shape
        image_center_x = width // 2
        
        #hsv for accuracy
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #masks for color detection
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        
        # reduces noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
       
        if contours:
          
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only consider contours with significant area
            if cv2.contourArea(largest_contour) > 100:
                ball_found = True
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate ball center
                ball_center_x = x + w // 2
                
                # Draw rectangle around the ball
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Calculate how far the ball is from center (in pixels)
                pixels_from_center = ball_center_x - image_center_x
                
                # Convert pixel distance to angle
                # Formula: angle = (pixel_distance / (width/2)) * (FOV/2)
                angle_to_turn = (pixels_from_center / (width/2)) * (camera_fov/2)
                
                # Display info on image
                text = f"Turn: {angle_to_turn:.1f} degrees {'right' if angle_to_turn < 0 else 'left'}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"Ball found! Turn {abs(angle_to_turn):.1f} degrees to the {'right' if angle_to_turn < 0 else 'left'}")
        
        cv2.imshow("test", frame)
        cv2.waitKey(100)
        
        img_name = f"imgcheck_{i}.png"
        cv2.imwrite(img_name, frame)  # Save the frame with the detection visualization
        print(f"{img_name} written!")
        
        if ball_found:
            # Save an additional image showing the successful detection
            detection_img = f"ball_detected_{i}.png"
            cv2.imwrite(detection_img, frame)
            print(f"Ball detected and saved as {detection_img}")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    return ball_found, angle_to_turn if ball_found else None