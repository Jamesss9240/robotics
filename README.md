# robotics

1. `pip install torch torchvision IPython pyttsx3 edge_tts stanza virtualenv pyaudio wave soundfile playsound faster_whisper pvporcupine opencv-python-headless`
2. If any other dependencies are missing that are stated pip install them too
3. `python -m venv myenv`
4. `source myenv/bin/activate`
5. run `stanza.download('en')` in a python script to download files for stanza
6. run `python check_for_block.py` make sure you are in the venv
7. say "hey vex find ball" or "hey vex hello" or "hey vex \[forward/backwards/left/right\] 200" 