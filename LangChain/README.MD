# LangChain Pipeline

This component implements the core generative AI pipeline that transforms sketches into detailed prompts and generates photorealistic images using LangChain, Ollama, and Stable Diffusion with ControlNet.

## Overview

The LangChain pipeline is responsible for:
- Processing user sketches and extracting semantic descriptions using language models
- Generating detailed prompts based on sketch descriptions and user stylistic instructions
- Orchestrating the image generation process with ControlNet
- Providing a REST API server for integration with the UI component

## Files and How to Use

### Core Files

- **Pipeline.py**: Main pipeline implementation
  - `SketchDescriptor`: Processes sketches and generates descriptions using Ollama language models
  - `PromptGenerator`: Creates detailed prompts from sketch descriptions and user instructions
  - `ImageGenerator`: Handles image generation with Stable Diffusion and ControlNet
  
  Usage: Import and instantiate the pipeline classes to build your processing workflow.

- **server.py**: REST API server for the pipeline
  - Exposes HTTP endpoints for sketch processing and image generation
  - Handles request/response serialization with JSON
  - Integrates with Traceloop for request tracing and monitoring
  
  Usage: Run with `python server.py` to start the API server on the configured port.

- **utils.py**: Utility functions and helper methods
  - Image preprocessing and formatting
  - Configuration loading utilities
  - Common helper functions used across the pipeline components
  

### Configuration Files

Configuration files are located in the `configs/` folder:

- **model_configs.json**: Contains model parameters and external service settings
  - Model parameters (temperature, top_p, etc.)
  - ControlNet and Stable Diffusion settings

- **model_instructions.json**: Detailed instructions for model execution
  - Prompt templates for sketch description and enhancement
    - System prompts for the language model

### How to Modify Model Configurations

1. Open the configuration files in `configs/`
2. Modify the parameters according to your project needs:
   - Update model paths if using different models
   - Adjust inference parameters (temperature etc) to control output quality and creativity
   - Customize prompts and instructions for different use cases
3. Save your changes and restart the server to apply the new configurations

### How to Modify Server Configurations
Create a `setup.env` file in the configs directory.

#### Ollama Server Configuration

To connect to a different Ollama server instance:

1. Open the `setup.env` file in the configs directory
2. Modify the `OLLAMA_SERVER` variable
3. Update the server address and port:
   ```env
   OLLAMA_SERVER="your-server-ip:11434"
   ```
4. Replace `your-server-ip` with your Ollama server hostname or IP address
5. Save the changes and restart the server

#### Traceloop Endpoint Configuration

To configure the Traceloop logging endpoint for request tracing:

1. Open the `setup.env` file in the configs directory
2. Modify the `TRACELOOP_ENDPOINT` variable
3. Update the endpoint URL:
   ```env
   TRACELOOP_ENDPOINT="http://your-traceloop-host:4318"
   ```
4. Replace `your-traceloop-host` with your Traceloop server hostname or IP address
5. Save the changes and restart the server

**Example setup.env file:**
```env
OLLAMA_SERVER="localhost:11434"
TRACELOOP_ENDPOINT="http://localhost:4318"
```

## Integration with Other Components

This LangChain component works in conjunction with:
- **UI/app.py**: Provides the Gradio-based user interface that calls this pipeline
- **Logging**: Monitors and logs pipeline execution through Traceloop
