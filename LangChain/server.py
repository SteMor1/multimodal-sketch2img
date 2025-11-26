from fastapi import FastAPI, UploadFile, Form
from Pipeline import run_pipeline
import tempfile

app = FastAPI()
@app.get("/health")
def health():
    return "ok"
@app.post("/generate")
async def generate(
    sketch: UploadFile,
    user_prompt: str = Form(...)
):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(await sketch.read())
    tmp.close()

    caption, final_prompt, img_b64 = run_pipeline(
        sketch_path=tmp.name,
        user_prompt=user_prompt
    )

    return {
        "caption": caption,
        "final_prompt": final_prompt,
        "image_base64": img_b64
    }

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="debug"
    )
