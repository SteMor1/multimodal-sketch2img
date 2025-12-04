#Utils
from utils import *

#Langchain
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaLLM
import base64
from io import BytesIO
from pydantic import BaseModel
# Env Loading
from dotenv import load_dotenv
import os

#Logging
from traceloop.sdk import Traceloop

#SD-ControlNet
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import os
#Pipeline
from langchain_core.runnables import RunnableSequence


###SETUP###
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # ONLY USE THIS LINE ON WINDOWS (COMMENT WHEN USING LINUX)

load_dotenv(dotenv_path="configs/setup.env",override=True)  
model_configs = load_model_configs()
#Infrastructure setup
OLLAMA_SERVER = os.getenv("OLLAMA_SERVER",default="localhost:11434")
TRACELOOP_ENDPOINT = os.getenv("TRACELOOP_ENDPOINT",default="localhost:4318")

#Models setup
prompt_creator_configs=model_configs["models"]["llm_prompt_creator"] 
sketch_descriptor_configs = model_configs["models"]["sketch_descriptor"] 
SKETCH_DESCRIPTOR_MODEL = sketch_descriptor_configs["name"]     #os.getenv("SKETCH_DESCRIPTOR_MODEL",default="None")
PROMPT_CREATOR_MODEL =  prompt_creator_configs["name"]    #os.getenv("PROMPT_CREATOR_MODEL",default="None")

# Setup resume
print(
    f"""
    #######INFRASTUCTURE#######
    Ollama Server: {OLLAMA_SERVER}
    Traceloop endpoint: {TRACELOOP_ENDPOINT}

    #######MODELS SETTINGS#######
    \tPrompt Creator 
    \t\tModel: {PROMPT_CREATOR_MODEL}
    \t\tTemperature: {prompt_creator_configs["temperature"]}
    \tSketch Descriptor 
    \t\tModel: {SKETCH_DESCRIPTOR_MODEL}
    \t\tTemperature: {sketch_descriptor_configs["temperature"]}
    """
)
### LOGGING ###

Traceloop.init(
    app_name="sketch-to-image-pipeline",
    api_endpoint=TRACELOOP_ENDPOINT 
)

#MODELS INSTANTATIATION

#Sketch Descriptor
sketch_descriptor_llm = ChatOllama(model=SKETCH_DESCRIPTOR_MODEL,base_url=OLLAMA_SERVER,temperature=sketch_descriptor_configs["temperature"]) #Multiple  Interaction
##Prompt creator
prompt_creator_llm = OllamaLLM(model=PROMPT_CREATOR_MODEL,base_url=OLLAMA_SERVER) #single interaction 
## Image generation
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
                         instruction
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