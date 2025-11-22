# %%
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# %%
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_lineart",
    torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda")


# %%
def load_sketch(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512, 512))
    img = cv2.Canny(img, 100, 200)
    img = Image.fromarray(img)
    return img


# %%
def generate_controlnet(sketch_path, prompt="a realistic photo"):
    sketch = load_sketch(sketch_path)

    result = pipe(
        prompt,
        image=sketch,
        num_inference_steps=20,
        guidance_scale=7.5
    )

    return result.images[0]


# %%
sketch_path = "sketch.jpg"

output = generate_controlnet(sketch_path, prompt="a colored realistic version of the sketch")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Sketch Input")
plt.imshow(Image.open(sketch_path))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("ControlNet Small Output")
plt.imshow(output)
plt.axis("off")

plt.show()



