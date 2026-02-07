# AIRA – Personal Voice-Based Small Language Model (SLM)

AIRA is a locally running, voice-based personal assistant powered by a fine-tuned Small Language Model (SLM).  
It is designed to run in the background and answer spoken questions using a custom-trained model.

The system runs fully on-device and uses:
- LoRA fine-tuning on a base language model
- Whisper for speech-to-text
- Offline text-to-speech
- No external APIs at runtime

---

## Project Structure

SLM/
│
├── data/
│ └── train.json
│
├── training/
│ └── train.py
│
├── slm/
│ ├── ayesha_slm/ # created after training
│ └── ayesha_slm.py
│
├── core/
│ ├── listener.py
│ ├── stt.py
│ ├── tts.py
│ └── brain.py
│
├── run_aira.py
├── requirements.txt
└── README.md

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
ffmpeg -version
python training/train.py
python run_aira.py
