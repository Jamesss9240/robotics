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
    
def pictest(images_folder="images"):
    #colour ranges in hsv
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 100, 100)
    upper_red2 = (180, 255, 255)
    
    
    lower_blue = (100, 100, 100)
    upper_blue = (130, 255, 255)
    
    
    camera_fov = 70
    KNOWN_BALL_DIAMETER = 70
    
    if not os.path.isdir(images_folder):
        print(f"Error: Folder '{images_folder}' does not exist.")
        return False, None
    
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Error: No images found in '{images_folder}'.")
        return False, None
    
    cv2.namedWindow("test")
    ball_found = False
    angle_to_turn = None
    
    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        height, width, channels = frame.shape
        image_center_x = width // 2
    
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        
        # reduce noise 
        kernel = np.ones((5, 5), np.uint8)

        mask_red = cv2.erode(mask_red, kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, kernel, iterations=2)

        mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
        
       
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_color = None
        ball_contour = None
        highlight_color = None
        
       
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 100:
                ball_color = "red"
                ball_contour = largest_red
                highlight_color = (0, 255, 0) 
        
       
        if contours_blue:
            largest_blue = max(contours_blue, key=cv2.contourArea)
           
            if cv2.contourArea(largest_blue) > 100 and (ball_contour is None or cv2.contourArea(largest_blue) > cv2.contourArea(ball_contour)):
                ball_color = "blue"
                ball_contour = largest_blue
                highlight_color = (0, 165, 255)  
        focal_length = (width / 2) / math.tan(math.radians(camera_fov / 2))
        if ball_contour is not None:
            ball_found = True
            
            M = cv2.moments(ball_contour)
            if M["m00"] != 0:
                ball_center_x = int(M["m10"] / M["m00"])
                ball_center_y = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(ball_contour)
                ball_center_x = x + w // 2
                ball_center_y = y + h // 2

            x, y, w, h = cv2.boundingRect(ball_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), highlight_color, 2)
            
            cv2.circle(frame, (ball_center_x, ball_center_y), 5, (255, 255, 255), -1)
            
            pixels_from_center = ball_center_x - image_center_x
            
            angle_to_turn = (pixels_from_center / (width/2)) * (camera_fov/2)
            turn_direction = 'right' if angle_to_turn > 0 else 'left'  
            
            
            distance = (KNOWN_BALL_DIAMETER * focal_length) / w
            if turn_direction == 'left':
                angle_to_turn = -abs(angle_to_turn) 
            else:
                angle_to_turn = abs(angle_to_turn)  
      
            text = f"{ball_color} ball: Turn {int(round(angle_to_turn))}° {turn_direction}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, highlight_color, 2)
            cv2.putText(frame, f"Distance: {distance:.0f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, highlight_color, 2)
            
            print(f"{ball_color.capitalize()} ball found in {image_file}! Width: {w}px, Turn {abs(angle_to_turn):.1f} degrees {turn_direction}, Distance: {distance:.0f}mm")
        
        cv2.imshow("test", frame)
        cv2.waitKey(1000)  
        
        # Save processed image
        output_dir = "processed_images"
        #os.makedirs(output_dir, exist_ok=True)
        #output_path = os.path.join(output_dir, f"processed_{image_file}")
        #cv2.imwrite(output_path, frame)
        #print(f"Processed image saved as {output_path}")
        
        if ball_found:
            
            angle_str = abs(angle_to_turn)  
            if turn_direction == 'left':
                angle_str = -angle_str
                
            detection_path = os.path.join(output_dir, f"turn_{angle_str}deg_{ball_color}_{image_file}")
            cv2.imwrite(detection_path, frame)
            print(f"{ball_color} ball detection saved as {detection_path}")
    
    cv2.destroyAllWindows()
    return ball_found, angle_to_turn if ball_found else None
pictest()