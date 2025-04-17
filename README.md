# robotics

1. `pip install torch torchvision IPython pyttsx3 edge_tts stanza virtualenv pyaudio wave soundfile playsound faster_whisper pvporcupine opencv-python-headless RPi.GPIO`
2. If any other dependencies are missing that are stated pip install them too
3. `python -m venv myenv`
4. `source myenv/bin/activate`
5. run `stanza.download('en')` in a python script to download files for stanza
6. run `python check_for_block.py` make sure you are in the venv
7. say "hey vex find ball" or "hey vex hello" or "hey vex \[forward/backwards/left/right\] 200"


Our robot, which can be used to pick up a block via voice, contains a Speech recognition system, that allows it to recognise commands through a microphone that is connected to a raspberry pi, allowing for the tasks the vex is programmed to do, which we will shortly mention, can be accessed and preformed fully via voice commands, with no need for any other interaction within the robot, we also have made a  Convolutional neural network model, that uses a hybrid system of model detection as well as basic hsv detection, this allows for the model to find the block without worrying about determing the specific colour of it, since the hsv values can reliably desquinguish the red from the blue block, the main reason for this approach is it asllows the model to have a higher accuraucy overall for finding the blocks due to the task at hand being less intensive than determining it from colour and the shaping of the block another final thing we have done with this robot is that we have been able to use the 3 wire interface on the vex and the GPIO pins on the PI to implement a custom bit banging wire protocol, which allows the pi to do the working to find the block with the model, and then send sequences of bytes to the vex at a rate of 50 bits/s, which allows the vex and pi to communicate, despite the proprietary nature of the vex itself

