# FACE-RECOGNITION-SYSTEM-USING-CNN-IN-DEEP-LEARNING-
This repository implements a Face Recognition System using Convolutional Neural Networks (CNN) in deep learning. It includes data preprocessing, model architecture design, and training using popular datasets. The system is capable of recognizing faces in images with high accuracy using CNN-based classifiers.

## Loki Desktop Voice Assistant
Loki is a modern, dark-themed desktop voice assistant with speech recognition, text-to-speech, app/website launching, and LLM-powered answers.

### Features
- Voice interaction (microphone input + TTS responses).
- System control for opening common apps and websites.
- LLM responses using OpenAI or Gemini.
- Futuristic GUI with live status and conversation history.

### Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure an LLM provider (choose one):
   - **OpenAI**
     - Create an API key at https://platform.openai.com/api-keys.
     - Set environment variables:
       ```bash
       export LOKI_LLM_PROVIDER=openai
       export LOKI_OPENAI_API_KEY="your_key_here"
       export LOKI_OPENAI_MODEL="gpt-4o-mini"
       ```
   - **Gemini**
     - Create an API key at https://aistudio.google.com/app/apikey.
     - Set environment variables:
       ```bash
       export LOKI_LLM_PROVIDER=gemini
       export LOKI_GEMINI_API_KEY="your_key_here"
       export LOKI_GEMINI_MODEL="gemini-1.5-pro"
       ```

### Run
```bash
python loki_assistant.py
```

### Usage Tips
- Click **Start Listening** to enable microphone capture.
- Say commands like:
  - “Open Chrome”
  - “Open Notepad”
  - “Open youtube.com”
  - “What’s the weather in Tokyo?”
