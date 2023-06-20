import os
import sys
import time



def look_gpu():
    while True:
        os.system("nvidia-smi")
        time.sleep(0.1)


# x=os.system("gnome-terminal"+" nvidia-smi")
# os.device_encoding(x).zfill("nvidia-smi")
look_gpu()


