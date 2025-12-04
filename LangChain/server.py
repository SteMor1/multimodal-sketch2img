from fastapi import FastAPI, UploadFile, Form
from Pipeline import run_pipeline
import tempfile
"""
server.py

This script implements a FastAPI server that provides two endpoints:

1. /health (GET): A health check endpoint that returns "ok".
2. /generate (POST): Accepts an uploaded sketch file and a user prompt, processes them
    using the `run_pipeline` function, and returns the generated caption, final prompt,
    and image in base64 format.

Dependencies:
- FastAPI: For building the API.
- tempfile: For creating temporary files to store the uploaded sketch.
- Pipeline: A custom module containing the `run_pipeline` function.

Usage:
Run the server with the following command:
     uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
- GET /health: Returns "ok" to indicate the server is running.
- POST /generate: Accepts a sketch file and user prompt, processes them, and returns the result.

"""
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
