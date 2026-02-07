# AIRA – Personal Voice-Based Small Language Model (SLM)

AIRA is a locally running, voice-based personal assistant powered by a fine-tuned Small Language Model (SLM).  
It is designed to run in the background and answer spoken questions using a custom-trained model.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
ffmpeg -version
python training/train.py
python run_aira.py
