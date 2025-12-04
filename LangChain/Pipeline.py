#Langchain
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaLLM
import base64
from io import BytesIO
from pydantic import BaseModel
# Env Loading
from dotenv import load_dotenv
import os
import json
#Logging
from traceloop.sdk import Traceloop
#SD-ControlNet
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import cv2
from PIL import Image
import numpy as np
import os
#Pipline
from langchain_core.runnables import RunnableSequence

### UTILS ###
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
    with open("model_instructions.json", "r", encoding="utf-8") as f:
        return json.load(f)
def load_model_configs():
        with open("model_configs.json", "r", encoding="utf-8") as f:
            return json.load(f)

###SETUP###
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # ONLY USE THIS LINE ON WINDOWS (COMMENT WHEN USING LINUX)

load_dotenv(dotenv_path="setup.env",override=True)  
model_configs = load_model_configs()

OLLAMA_SERVER = os.getenv("OLLAMA_SERVER",default="localhost:11434")
TRACELOOP_ENDPOINT = os.getenv("TRACELOOP_ENDPOINT",default="localhost:4318")

PROMPT_CREATOR_MODEL = os.getenv("PROMPT_CREATOR_MODEL",default="None")
SKETCH_DESCRIPTOR_MODEL = os.getenv("SKETCH_DESCRIPTOR_MODEL",default="None")
print(f"Ollama Server: {OLLAMA_SERVER}\nPrompt Creator Model: {PROMPT_CREATOR_MODEL}\nSketch Descriptor Model: {SKETCH_DESCRIPTOR_MODEL}")
### LOGGING ###

Traceloop.init(
    app_name="sketch-to-image-pipeline",
    api_endpoint="http://stefano-ideapad-pro-5:4318" #TODO add parameter for tempo endpoint
 
)

#Model Instatiation
sketch_descriptor_llm = ChatOllama(model=SKETCH_DESCRIPTOR_MODEL,base_url=OLLAMA_SERVER,temperature=0) #Multiple  Interaction
prompt_creator_llm = OllamaLLM(model=PROMPT_CREATOR_MODEL,base_url=OLLAMA_SERVER) #single interaction 


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda")


### TOOLS DEFINITION ###

## Sketch descriptor
class SketchDescInput(BaseModel):
    image_path: str

@tool("get_sketch_desc_prompt", args_schema=SketchDescInput)
def get_sketch_desc_prompt(image_path: str) -> str:
    """
    Get a description for the sketch
    
    Expected input:
    {
        "image_path": "...",
       
    }
    """
    
    instruction = load_instructions()["sketch_descriptor_instruction"]
    
    image_b64 = encode_image(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "SYSTEM RULES (FOLLOW STRICTLY):\n"+ instruction
                    )
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_b64}"
                }
            ]
        }
    ]




    result = sketch_descriptor_llm.invoke(messages)
    return result.content.strip()
# Prompt Creation
prompt_creator_instruction_template="""You are a prompt engineer for image diffusion models (Stable Diffusion + ControlNet).

Your task is to combine:
1) the sketch caption (describing the structure and pose extracted from the drawing)
2) the user instruction (stylistic preferences or artistic intent)

CAPTION:
"{caption}"

USER REQUEST:
"{user_prompt}"

RULES:
- Respect EXACTLY the shape, pose, and structure described in the caption.
- Integrate the user's request ONLY in stylistic or atmospheric ways.
- NEVER override or contradict the sketch caption.
- Add photorealistic details (lighting, lens, textures) unless the user asks for a non-realistic style.
- Produce ONE fluent sentence.
- DO NOT show chain-of-thought.
- Output ONLY the final prompt.
"""
class PromptCreatorInput(BaseModel):
    caption: str
    user_prompt:str

@tool("create_sd_prompt",args_schema=PromptCreatorInput)
def create_sd_prompt(    caption: str, user_prompt:str) -> str:
    """
    Combine the sketch caption and the user prompt into a
    photorealistic Stable Diffusion / ControlNet prompt.
    
    Expected input:
    {
        "caption": "...",
        "user_prompt": "..."
    }
    """
    instruction = load_instructions()["prompt_creator_instruction_template"]
    instruction = instruction.format(
    caption=caption,
    user_prompt=user_prompt)


    result = prompt_creator_llm.invoke(instruction)
    return result.strip()

#  Image Generation
class SdGeneratorInput(BaseModel):
    sketch_path: str
    prompt:str
@tool("generate_sd_image",args_schema=SdGeneratorInput)
def generate_controlnet(sketch_path:str, prompt:str):
    """Tool to generate an image using sd given an input sketch and a prompt"""
    sketch = load_sketch(sketch_path)

    controlnet_params = load_model_configs()["models"]["controlnet-sd"] # load parameters for each request to enable more control over the pipline
    negative_prompt = controlnet_params["negative_prompt"]
    num_inference_step = controlnet_params["num_inference_steps"]
    guidance_scale = controlnet_params["guidance_scale"]
    controlnet_conditioning_scale = controlnet_params["controlnet_conditioning_scale"]

    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=sketch,
        num_inference_steps=num_inference_step,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    
    )

    return result.images[0]
### PIPLINE DEFINTION ###


pipeline = RunnableSequence(
    # Step 1 — pass inputs through
    lambda inputs: inputs,

    # Step 2 — captioning
    lambda inputs: {
        "caption": get_sketch_desc_prompt.invoke(
            {"image_path": inputs["image_path"]}
        ),
        "user_prompt": inputs["user_prompt"],
        "sketch_path": inputs["image_path"]
    },

    # Step 3 — prompt generation
    lambda inputs: {
        "caption": inputs["caption"],
        "final_prompt": create_sd_prompt.invoke({
            "caption": inputs["caption"],
            "user_prompt": inputs["user_prompt"]
        }),
        "sketch_path": inputs["sketch_path"]
    },

    # Step 4 — SD image generation 
    lambda inputs: {
        "caption": inputs["caption"],
        "final_prompt": inputs["final_prompt"],
        "generated_image": generate_controlnet.invoke({
            "sketch_path": inputs["sketch_path"],
            "prompt": inputs["final_prompt"]
        }),
    }
)

def run_pipeline(sketch_path: str, user_prompt: str):
    """
    Full pipeline:
    1. Describe sketch
    2. Create final SD prompt
    3. Generate image with ControlNet
    4. Return (caption, final_prompt, base64_image)
    """

    result = pipeline.invoke({
        "image_path": sketch_path,
        "user_prompt": user_prompt})
    buffer = BytesIO()
    result["generated_image"].save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    caption = result["caption"]
    final_prompt = result["final_prompt"]
    return caption, final_prompt, img_b64