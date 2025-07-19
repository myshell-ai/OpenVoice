from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import os
import torch
import langid
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import shutil

app = FastAPI()

# Configuration from openvoice_app.py
en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)

supported_languages = ['zh', 'en']

@app.get("/")
async def root():
    return {"message": "Welcome to the OpenVoice API! Server is up and running!"}

@app.post("/synthesize/")
async def synthesize_speech(
    prompt: str = Form(...),
    style: str = Form(...),
    audio_file: UploadFile = File(...),
):

    # Save the uploaded audio file temporarily
    temp_audio_path = os.path.join(output_dir, audio_file.filename)
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    language_predicted = langid.classify(prompt)[0].strip()
    print(f"Detected language: {language_predicted}")

    if language_predicted not in supported_languages:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=400, detail=f"The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}")

    if language_predicted == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = 'Chinese'
        if style not in ['default']:
            os.remove(temp_audio_path)
            raise HTTPException(status_code=400, detail=f"The style {style} is not supported for Chinese, which should be in ['default']")
    else:
        tts_model = en_base_speaker_tts
        if style == 'default':
            source_se = en_source_default_se
        else:
            source_se = en_source_style_se
        language = 'English'
        if style not in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']:
            os.remove(temp_audio_path)
            raise HTTPException(status_code=400, detail=f"The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']")

    if len(prompt) < 2:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=400, detail="Please give a longer prompt text")
    if len(prompt) > 200:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=400, detail="Text length limited to 200 characters for this demo, please try shorter text.")

    try:
        target_se, audio_name = se_extractor.get_se(temp_audio_path, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Get target tone color error: {str(e)}")

    src_path = os.path.join(output_dir, 'tmp.wav')
    tts_model.tts(prompt, src_path, speaker=style, language=language)

    save_path = os.path.join(output_dir, 'output.wav')
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message
    )

    # Clean up temporary files
    os.remove(temp_audio_path)
    os.remove(src_path)

    return FileResponse(save_path, media_type="audio/wav", filename="synthesized_audio.wav")

