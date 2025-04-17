import math
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import RPi.GPIO as GPIO
import time
import pyaudio
import wave
from IPython.display import display, Audio
import soundfile as sf
from playsound import playsound
import asyncio
import struct
import pvporcupine
import edge_tts
from faster_whisper import WhisperModel
import stanza

com_stop = 0
com_forward = 1
com_reverse = 2
com_turn_left = 3
com_turn_right = 4
com_turn_left_angle = 5
com_turn_right_angle = 6
com_forward_for = 7
com_reverse_for = 8
com_arm_up_for = 9
com_arm_down_for = 10
com_arm_up = 11
com_arm_down = 12
com_claw_open = 13
com_claw_close = 14
com_drive_velocity = 15
com_claw_velocity = 16
com_arm_velocity = 17
com_arm_position = 18

def comm_setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(23, GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(24, GPIO.OUT,initial=GPIO.LOW)
    GPIO.output(23,GPIO.LOW)
    GPIO.output(24,GPIO.LOW)
    print("comm setup")

def send_data(data):
    buf = bytes(data)
    bits_sent = 0
    b_mask = 1
    print("sending: ", buf)
    for i in range(len(buf)):
        for b in range(8):
            b_mask = 1 << b
            s = buf[i] & b_mask # the bit to send
            if s:
                GPIO.output(24,GPIO.HIGH)
            else:
                GPIO.output(24,GPIO.LOW)
            time.sleep(0.015) # 15ms
            GPIO.output(23,GPIO.HIGH) # set clock high
            time.sleep(0.015) # 15ms
            GPIO.output(23,GPIO.LOW) # set clock low
            bits_sent += 1
        b_mask = 1 # reset bitmask

def encode_num(n):
    return n.to_bytes()

comm_setup()

def take_picture():

    cap = cv2.VideoCapture(0)


    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        return None

    
    #save to images folder
    images_folder = "images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    image_file = os.path.join(images_folder, "captured_image_"+ str(time.time_ns())+".jpg")
    cv2.imwrite(image_file, frame)
    print(f"Image saved as {image_file}")
    return frame, image_file
    


    
def pictest():
    camera_fov = 90
    KNOWN_BALL_DIAMETER = 100

    ball_found = False
    angle_to_turn = None

    option = 1 

    #model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './cnn_hsv_hybrid.pth'  # Path to the hybrid model
    
    try:
        
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        
       #extract model parameters
        model_state_dict = model_data['model_state_dict']
        hsv_params = model_data['hsv_params']
        colour_mapping = model_data.get('colour_mapping', {})
      
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
                #
                self.attention = nn.Sequential(
                    nn.Conv2d(32, 1, kernel_size=1),
                    nn.Sigmoid()
                )
                self.fc = nn.Linear(8*8*32, 4)
                
            def forward(self, x):
                #make sure the batch id is of correct number of dimensionms
                if x.dim() == 3:
                    x = x.unsqueeze(0) 
                
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)
                return out
                
        #reload model
        model = ConvNet().to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print("model loaded successfully")
    except Exception as e:
        #if model cant load, rely on basic hsv detection (unreliable)
       
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
    
    #process image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def hybrid_prediction(image, hsv_params, model=None):
        #convert to compatable format
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColour(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
            image = cv2.cvtColour(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        
        hsv_img = cv2.cvtColour(image, cv2.COLOR_BGR2HSV)
        
        #create colour masks
        mask_red1 = cv2.inRange(hsv_img, hsv_params['red']['lower1'], hsv_params['red']['upper1'])
        mask_red2 = cv2.inRange(hsv_img, hsv_params['red']['lower2'], hsv_params['red']['upper2'])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
    
        mask_blue = cv2.inRange(hsv_img, hsv_params['blue']['lower'], hsv_params['blue']['upper'])
        
        #reduces noise, improves detection capability
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.erode(mask_red, kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, kernel, iterations=2)
        mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
        
        # helper for finding boundaries of object
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
      
        largest_red = max(contours_red, key=cv2.contourArea, default=None) if contours_red else None
        largest_blue = max(contours_blue, key=cv2.contourArea, default=None) if contours_blue else None
        
        hsv_colour = 'none'
        hsv_confidence = 0.0
        hsv_contour = None
        
        # use the hsv to  detirimine colour
        red_area = cv2.contourArea(largest_red) if largest_red is not None else 0
        blue_area = cv2.contourArea(largest_blue) if largest_blue is not None else 0
        
        if red_area > blue_area and red_area > 100:
            hsv_colour = 'red'
            hsv_confidence = red_area / (image.shape[0] * image.shape[1])
            hsv_contour = largest_red
        elif blue_area > red_area and blue_area > 100:
            hsv_colour = 'blue'
            hsv_confidence = blue_area / (image.shape[0] * image.shape[1])
            hsv_contour = largest_blue
        
        #use the cnn to reinforce this, and correct errors that the hsv prediction may have got
        cnn_colour = 'none'
        cnn_confidence = 0.0
        
        if model is not None:
            try:
                #conv image to tensor
                image_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    _, predicted = torch.max(outputs, 1)
                    predicted_idx = predicted.item()
                    
                    #mapping for predictions
                    if predicted_idx == 0:
                        cnn_colour = 'blue' 
                    elif predicted_idx == 2:  
                        cnn_colour = 'red'  
                    else:
                        cnn_colour = 'none'
                        
                    cnn_confidence = probabilities[predicted_idx].item() #confidence
            except Exception as e:
                print(f"Error in CNN prediction")

        final_colour = 'none'
        final_contour = None
        
        if hsv_confidence > 0.05:  
            final_colour = hsv_colour
            final_contour = hsv_contour
        elif cnn_confidence > 0.7:  
            final_colour = cnn_colour
          
            if final_colour == 'red' and largest_red is not None:
                final_contour = largest_red
            elif final_colour == 'blue' and largest_blue is not None:
                final_contour = largest_blue
        
        return final_colour, final_contour, cnn_confidence
    
   
    for i in range(1):
        if option == 1:
            #take picture
            try:
                frame, image_file = take_picture()
                if frame is None:
                    print("Failed to capture image")
                    continue
            except Exception as e:
                print(f"Error capturing image: {e}")
                continue
        else:
            if i == 0: 
                try:
                    images_folder = "images"
                    if not os.path.exists(images_folder):
                        print(f"Images folder not found")
                        os.makedirs(images_folder)
                        print("add images inside the folder and try again.")
                        break
                        
                    image_files = [f for f in os.listdir(images_folder) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    
                    if not image_files:
                        print(f"No image files found in {images_folder}. Please add some images.")
                        break
                        
                    print(f"Found {len(image_files)} images. Processing first image")
                    image_file = image_files[0]
                    image_path = os.path.join(images_folder, image_file)
                    
                    print(f"Loading image: {image_path}")
                    frame = cv2.imread(image_path)
                    
                    if frame is None:
                        print("Failed to load image.")
                        break
                        
                except Exception as e:
                    print(f"Error loading image: {e}")
                    break
            else:
               
                break
        
        if frame is None:
            print("No valid frame to process")
            continue
       
        height, width, channels = frame.shape
        image_center_x = width // 2
        
      
        ball_colour, ball_contour, cnn_confidence = hybrid_prediction(frame, hsv_params, model)
        
     
        ball_found = ball_contour is not None and ball_colour != 'none'
        

        display_frame = frame.copy()
        
        if ball_found:
        
            highlight_colour = (0, 255, 0) if ball_colour == "red" else (0, 165, 255) 
            
            
            M = cv2.moments(ball_contour)
            if M["m00"] != 0:
                ball_center_x = int(M["m10"] / M["m00"])
                ball_center_y = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(ball_contour)
                ball_center_x = x + w // 2
                ball_center_y = y + h // 2
            
            # draw rectangle around the ball
            x, y, w, h = cv2.boundingRect(ball_contour)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), highlight_colour, 2)
            
            # draw centre
            cv2.circle(display_frame, (ball_center_x, ball_center_y), 5, (255, 255, 255), -1)
            
            # turn needed calculation
            focal_length = (width / 2) / math.tan(math.radians(camera_fov / 2))
            pixels_from_center = ball_center_x - image_center_x
            angle_to_turn = (pixels_from_center / (width/2)) * (camera_fov/2) /3
            print("angle to turn: ", angle_to_turn)
            turn_direction = 'right' if angle_to_turn > 0 else 'left'
            if turn_direction == 'right':
                print("turn right")
                command = bytearray()
                command.append(0x06)
                r_angle = int(abs(round(angle_to_turn)))
                print(r_angle)
                command.extend(r_angle.to_bytes(2, byteorder='big', signed=False))
                send_data(command)
            if turn_direction == 'left':
                print("turn left")
                command = bytearray()
                command.append(0x05)
                r_angle = int(abs(round(angle_to_turn)))
                print(r_angle)
                command.extend(r_angle.to_bytes(2, byteorder='big', signed=False))
                send_data(command)
            time.sleep(3)
            #calculate distance
            distance = ((KNOWN_BALL_DIAMETER * focal_length) / w) * 2
            print("round dist", round(distance))
            command = bytearray()
            command.append(com_claw_open)
            send_data(command)
            time.sleep(3)
            command = bytearray()
            command.append(com_forward_for)
            command.extend(round(distance).to_bytes(2, byteorder='big', signed=False))
            send_data(command)
            time.sleep(3)
            command = bytearray()
            command.append(com_claw_close)
            send_data(command)
            time.sleep(3)
            command = bytearray()
            command.append(com_arm_up)
            send_data(command)
            time.sleep(3)
            command = bytearray()
            command.append(com_stop)
            send_data(command)

            if turn_direction == 'left':
                angle_to_turn = -abs(angle_to_turn)
            else:
                angle_to_turn = abs(angle_to_turn)
            
            #adds text info to image for debugging
            text = f"{ball_colour} ball: Turn {int(round(angle_to_turn))}° {turn_direction} confidence: = {cnn_confidence}"
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, highlight_colour, 2)
            cv2.putText(display_frame, f"Distance: {distance:.0f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, highlight_colour, 2)
            
            print(f"{ball_colour.capitalize()} ball found in {image_file}! Width: {w}px, Turn {abs(angle_to_turn):.1f} degrees {turn_direction}, Distance: {distance:.0f}mm")
            #save image
            output_dir = "processed_images"
            os.makedirs(output_dir, exist_ok=True)
            
         
            angle_str = f"{int(round(angle_to_turn))}"
            
            detection_path = os.path.join(output_dir, f"turn_{angle_str}deg_{ball_colour}_{image_file}")
            cv2.imwrite(detection_path, display_frame)
            print(f"{ball_colour} ball detection saved as {detection_path}")
        else:
            cv2.putText(display_frame, "No ball detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"No ball detected in {image_file}")

    return ball_found, angle_to_turn if ball_found else None, ball_colour if ball_found else None

# Configuration
CHUNK = 1024                   # Number of audio samples per frame
FORMAT = pyaudio.paInt16       # Audio format (16-bit PCM)
CHANNELS = 1                   # Mono audio, don't need no fancy stero round 'ere
RATE = 16000                   # Audio sample rate (Hz)
SILENCE_THRESHOLD = 700        # Amplitude threshold for silence detection
MAX_SILENCE_CHUNKS = 30        # Number of small-volume chunks to consider as silence
WAKE_KEYWORD = "hey vex"       # Activation keyword

# Audio recording and saving
def record_command_audio():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Start speaking your command…")
    frames = []
    silence_chunks = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        # Compute volume for silence detection.
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        volume = np.max(np.abs(audio_chunk))
        if volume < SILENCE_THRESHOLD:
            silence_chunks += 1
        else:
            silence_chunks = 0

        if silence_chunks > MAX_SILENCE_CHUNKS:
            print("Silence detected. Finishing recording.")
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()
    return b"".join(frames)

def save_wave(filename, audio_data, channels=CHANNELS, rate=RATE, fmt=FORMAT):
    pa = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pa.get_sample_size(fmt))
    wf.setframerate(rate)
    wf.writeframes(audio_data)
    wf.close()

# Audio transcription
whisper_model = "tiny" #"small"
def transcribe_audio(audio_filename):
    print("Transcribing audio…")
    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_filename, beam_size=5)
    transcribed_text = "".join(segment.text for segment in segments)
    print("Transcription result:", transcribed_text)
    return transcribed_text

# Edge Text-to-speech
voice = "en-GB-SoniaNeural"
async def generate_tts_response(response_text, output_file="response.mp3"):
    print("Generating TTS response…")
    communicate = edge_tts.Communicate(response_text, voice)
    await communicate.save(output_file)
    print(f"TTS response saved as '{output_file}'.")

nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=False)

def detect_commands(text):
    doc = nlp(text)
    detected_commands = {}
    action = None
    direction = None
    value = None
    # We'll iterate over each sentence in case the text has multiple instructions.
    for sentence in doc.sentences:
        for word in sentence.words:
            token_text = word.text.lower()
            token_lemma = word.lemma.lower()

            # Look for action verbs
            if token_lemma in ["move", "go"]:
                action = token_lemma

            # Look for directional keywords
            if token_text in ["forward", "forwards", "backward", "backwards", "reverse", "turn left", "turn right", "left","right"]:
                direction = token_text

            # Look for numeric tokens
            if word.pos == "NUM" or token_text.isdigit():
                try:
                    value = int(token_text)
                except ValueError:
                    value = None
                    pass
    return direction, value

# voice assistant main loop
async def main():
    # Set up wake-word detection using Porcupine
    try:
        porcupine = pvporcupine.create(access_key="9UdYOMoMj15nqt24CfMgXOIQl2pA6mk+nKi0Wy9vnIf+EQfVQBqflg==",keyword_paths=["./Hey-Vex_en_raspberry-pi_v3_0_0.ppn"])#keywords=[WAKE_KEYWORD])
    except Exception as e:
        print("Error initializing Porcupine wake word engine:", e)
        return

    pa = pyaudio.PyAudio()
    wake_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print(f"Voice Assistant is online. Say '{WAKE_KEYWORD}' to activate.")

    try:
        while True:
            # Read audio frames for wake-word detection.
            pcm = wake_stream.read(porcupine.frame_length)
            pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm_unpacked)

            if keyword_index >= 0:
                print("Activation keyword detected!")
                try:
                    playsound("digital-bleep.mp3")
                except Exception as e:
                    print("Error during playback:", e)
                understood = False
                # Record the user’s command after activation.
                audio_data = record_command_audio()
                input_filename = "input.wav"
                save_wave(input_filename, audio_data)
                
                # Transcribe the recorded command.
                transcribed_text = transcribe_audio(input_filename)
                transcribed_text = transcribed_text.lower().strip() # strip spaces around text
                transcribed_text = transcribed_text.replace(".", "").replace("?", "").replace("!", "") # remove punctuation
                if transcribed_text == "hello" or transcribed_text == "hello there":
                    understood = True
                    response_text = "Hello I am vex, nice to meet you"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                if transcribed_text == "translation" or transcribed_text == "small model":
                    understood = True
                    print("set small model")
                    whisper_model = "small"
                    response_text = "set to small model"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                if transcribed_text == "tiny model":
                    understood = True
                    print("set tiny model")
                    whisper_model = "tiny"
                    response_text = "set to tiny model"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                if transcribed_text == "base model":
                    understood = True
                    print("set base model")
                    whisper_model = "base"
                    response_text = "set to base model"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                if transcribed_text == "large model":
                    understood = True
                    print("set large model")
                    whisper_model = "large"
                    response_text = "set to large model"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                if transcribed_text == "find ball" or transcribed_text == "pickup ball" or transcribed_text == "pick up the ball" or transcribed_text == "find the ball" or transcribed_text == "get the ball":
                    understood = True
                    print("running pictest")
                    response_text = "Trying to find and pickup ball"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                    b, a, c = pictest()
                    if not b:
                        response_text = "Could not find ball"
                        await generate_tts_response(response_text)
                        try:
                            playsound("response.mp3")
                        except Exception as e:
                            print("Error during playback:", e)
                    else:
                        if c == "blue":
                            response_text = "Found blue balls"
                            await generate_tts_response(response_text)
                            try:
                                playsound("response.mp3")
                            except Exception as e:
                                print("Error during playback:", e)
                        elif c == "red":
                            response_text = "Found red balls"
                            await generate_tts_response(response_text)
                            try:
                                playsound("response.mp3")
                            except Exception as e:
                                print("Error during playback:", e)

                if transcribed_text == "claw open" or transcribed_text == "claw unclamp" or transcribed_text == "grabber open" or transcribed_text == "do not grab":
                    understood = True
                    print("claw open")
                    response_text = "Opening claw"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                    command = bytearray()
                    command.append(com_claw_open)
                    send_data(command)
                if transcribed_text == "claw close" or transcribed_text == "claw clamp" or transcribed_text == "grabber close" or transcribed_text == "grab":
                    understood = True
                    print("claw close")
                    response_text = "Closing claw"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                    command = bytearray()
                    command.append(com_claw_close)
                    send_data(command)
                if transcribed_text == "arm up" or transcribed_text == "lift arm" or transcribed_text == "armature up":
                    understood = True
                    print("arm up")
                    response_text = "Lifting arm"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                    command = bytearray()
                    command.append(com_arm_up)
                    send_data(command)
                if transcribed_text == "arm down" or transcribed_text == "lower arm" or transcribed_text == "armature down":
                    understood = True
                    print("arm down")
                    response_text = "Lowering arm"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)
                    command = bytearray()
                    command.append(com_arm_down)
                    send_data(command)

                d, v = detect_commands(transcribed_text)
                if v is not None:
                    understood = True
                    if d == "forward" or d == "forwards":
                        print("forward")
                        response_text = "Moving forward"
                        await generate_tts_response(response_text)
                        try:
                            playsound("response.mp3")
                        except Exception as e:
                            print("Error during playback:", e)
                        command = bytearray()
                        command.append(com_forward_for)
                        command.extend(v.to_bytes(2, byteorder='big', signed=False))
                        send_data(command)
                    if d == "reverse" or d == "backwards" or d == "backward":
                        print("backward")
                        response_text = "Moving backwards"
                        await generate_tts_response(response_text)
                        try:
                            playsound("response.mp3")
                        except Exception as e:
                            print("Error during playback:", e)
                        command = bytearray()
                        command.append(com_reverse_for)
                        command.extend(v.to_bytes(2, byteorder='big', signed=False))
                        send_data(command)
                    if d == "turn left" or d == "left":
                        print("turn left")
                        response_text = "Turning left"
                        await generate_tts_response(response_text)
                        try:
                            playsound("response.mp3")
                        except Exception as e:
                            print("Error during playback:", e)
                        command = bytearray()
                        command.append(com_turn_left_angle)
                        if v > 360:
                            v = 360
                        if v < 0:
                            v = 0
                        command.extend(v.to_bytes(2, byteorder='big', signed=False))
                        send_data(command)
                    if d == "turn right" or d == "right":
                        print("turn right")
                        response_text = "Turning right"
                        await generate_tts_response(response_text)
                        try:
                            playsound("response.mp3")
                        except Exception as e:
                            print("Error during playback:", e)
                        command = bytearray()
                        command.append(com_turn_right_angle)
                        if v > 360:
                            v = 360
                        if v < 0:
                            v = 0
                        command.extend(v.to_bytes(2, byteorder='big', signed=False))
                        send_data(command)
                    if d == "arm":
                        command = bytearray()
                        command.append(com_arm_posistion)
                        if v < 40:
                            v = 40
                        if v > 450:
                            v = 450
                        command.extend(v.to_bytes(2, byteorder='big', signed=False))
                        send_data(command)
                if not understood:
                    print("did not understand command")
                    response_text = "Sorry I did not understand"
                    await generate_tts_response(response_text)
                    try:
                        playsound("response.mp3")
                    except Exception as e:
                        print("Error during playback:", e)

                print("Cycle complete. Awaiting next activation…")
    except KeyboardInterrupt:
        print("\nVoice assistant terminated by user.")
    finally:
        wake_stream.stop_stream()
        wake_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    asyncio.run(main())


