# UI
A simple UI that enables the user to access the Sketch to image system.

## How to use it
1. Create the environment with `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate sketch-to-image-client
   ```
3. Update the server URL in `config.yaml` to point to your server:
   ```yaml
   server_url: "http://your-server-url:port"
   ```
4. Launch **app.py**:
   ```bash
   python app.py
   ```

## Requirements
- Ensure you have Conda installed on your system.
