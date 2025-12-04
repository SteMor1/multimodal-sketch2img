# Sketch to Image project

This repository contains a generative AI pipeline designed to transform user-provided sketches into photorealistic images. The core of the project is a modular pipeline that integrates multiple tools and models to achieve this goal.

## Pipeline Overview

The pipeline consists of the following steps:

1. **Sketch Description**: Extracts a semantic description of the sketch using a language model.
2. **Prompt Creation**: Combines the sketch description with user-provided stylistic instructions to generate a detailed prompt for image generation.
3. **Image Generation**: Utilizes Stable Diffusion with ControlNet to generate a photorealistic image based on the sketch and the generated prompt.
4. **Output**: Returns the sketch description, final prompt, and the generated image.

## Key Components

- **LangChain/Pipeline.py**: Implements the pipeline logic, including sketch preprocessing, prompt generation, and image synthesis.
- **LangChain/server.py**: Provides an API endpoint to interact with the pipeline.
- **UI/app.py**: A Gradio-based user interface for sketch-to-image generation.

## How to Run

1. Set up the conda environment using the provided `environment.yml` file.
2. Start the logging components by using the docker compose file inside the `Logging` folder
3. Start the server by running `server.py`.
4. Optionally, launch the Gradio UI for an interactive experience.

For detailed instructions, refer to the respective component files.