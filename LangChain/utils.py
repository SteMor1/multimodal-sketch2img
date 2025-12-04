import json
import base64
import cv2
from PIL import Image

"""
This module provides utility functions for encoding images, loading sketches, 
and reading configuration files for a generative AI project.
Functions:
----------
- encode_image(path: str) -> str:
    Encodes an image file into a base64 string format.
- load_sketch(path):
    Loads an image as a grayscale sketch, resizes it to 512x512 pixels, 
    and converts it into a PIL Image object.
- load_instructions():
    Reads and returns the content of the "configs/model_instructions.json" file 
    as a Python dictionary.
- load_model_configs():
    Reads and returns the content of the "configs/model_configs.json" file 
    as a Python dictionary.
"""

def encode_image(path: str) -> str:
    """Encode image to base64"""
    #Sketch descriptor wants this format
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8") 
    
def load_sketch(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512, 512))
    #img = cv2.Canny(img, 250 ,250)
    img = Image.fromarray(img)
    return img
def load_instructions():
    with open("configs/model_instructions.json", "r", encoding="utf-8") as f:
        return json.load(f)
def load_model_configs():
        with open("configs/model_configs.json", "r", encoding="utf-8") as f:
            return json.load(f)