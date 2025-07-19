## Local Development Setup

Follow these steps to set up and run the application locally for development and debugging.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/namanthapliyal/OpenVoice.git
cd ./OpenVoice
```

### 2. Build the Docker Image

In the root location of the project, build the Docker image using the following command:

```bash
docker build -t openvoice-fastapi .
```

This command will:

Pull the nvidia/cuda base image.
Install necessary system dependencies and Python packages.
Clone the OpenVoice library.
Download pre-trained checkpoints required for voice synthesis.
Set up the working directory and expose the application port.
Build the Docker image.

This process may take some time, especially during the initial download of the base image and checkpoints.

### 3. Run the Docker Container

Once the image is built, you can run a container from it. To enable GPU acceleration and map the application's port to your host machine, use the following command:

```bash
docker run --gpus all -p 7860:7860 openvoice-fastapi

```

- --gpus all: Exposes all available NVIDIA GPUs on your host to the container. Ensure the NVIDIA Container Toolkit is correctly installed.
- -p 7860:7860: Maps port 7860 inside the container (where FastAPI runs) to port 7860 on your host machine.

The FastAPI application will now be accessible at http://localhost:7860.

### 4. Interact with the API

You can test the API using curl or any API client (like Postman, Insomnia, or your browser for GET requests). The primary endpoint is /synthesize/ which accepts POST requests with multipart/form-data.

Example curl Request:

```bash
curl -X POST "http://localhost:7860/synthesize/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=This is a test sentence for voice synthesis." \
  -F "style=default" \
  -F "audio_file=@/path/to/your/reference_audio.mp3" \
  -F "agree=true" \
  --output synthesized_audio.wav
```

Parameters:

- prompt (string, required): The text to be synthesized.
- style (string, required): The speaking style. Supported values: default, whispering, shouting, excited, cheerful, terrified, angry, sad, friendly. (Note: Chinese only supports default).
- audio_file (file, required): An audio file (.mp3 or .wav) of the reference speaker whose voice you want to clone.
- agree (boolean, required): Must be true to accept the terms and conditions.

The API will return the synthesized audio as a .wav file.

Output Directory
Synthesized audio files and temporary processing files will be stored in the outputs/ directory within the container. For local debugging, you might want to mount a volume to persist these outputs on your host machine.

### 4. Access Swagger Doc

You can access the Swagger UI documentation by navigating to http://localhost:7860/docs in your web browser. This provides an interactive API reference and allows you to test the API endpoints directly through the UI.
