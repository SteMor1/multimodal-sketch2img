import gradio as gr
import requests
import tempfile
import base64
from PIL import Image
from io import BytesIO

API_URL = "http://desktop-roma:8000/generate"

def process(sketch_img, user_prompt):
    if sketch_img is None:
        return "No sketch provided", "", None

    # salva sketch come file temporaneo
    print(sketch_img.keys())
    if isinstance(sketch_img, dict):
        sketch_img = sketch_img["composite"]
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=True)
    print(temp.name)
    sketch_img.save(temp.name)

    # invia al server remoto
    with open(temp.name, "rb") as f:
        response = requests.post(
            API_URL,
            files={"sketch": ("sketch.png", f, "image/png")},
            data={"user_prompt": user_prompt}
        ).json()

    caption = response["caption"]
    final_prompt = response["final_prompt"]

    # decodifica base64 in PIL Image
    output_b64 = response["image_base64"]
    img_data = base64.b64decode(output_b64)
    img = Image.open(BytesIO(img_data))

    return caption, final_prompt, img


with gr.Blocks() as demo:
    gr.Markdown("# 🎨→📸 Sketch-to-Image Generator")

    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(label="Draw your sketch",layers=False,type="pil" ,height=512 )
        with gr.Column():
            output_img = gr.Image(label="Generated Image",height=512 )
    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(label="Your prompt")
            btn = gr.Button("Generate")

    caption_box = gr.Textbox(label="Sketch description", interactive=False,lines=3) # Label for the generated caption
    final_prompt_box = gr.Textbox(label="Final prompt used", interactive=False,lines=3) #Label for the generated final prompt

    btn.click(process, [sketch, user_prompt],
              [caption_box, final_prompt_box, output_img])

demo.launch()
