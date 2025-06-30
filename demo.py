import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
#from openvoice.api import ToneColorConverter
from melo.api import TTS
from datetime import datetime

def get_openvoice_path():
    return '/home/tim/OpenVoice'

def get_ckpts_path():
    return f'{get_openvoice_path()}/checkpoints_v2/'

def get_ckpt_converter_path():
    return f'{get_ckpts_path()}converter'

def get_output_dir():
    return '/home/tim/OpenVoice/outputs_v2'

tone_color_converter = None

def get_tone_color_converter():
    global tone_color_converter
    if tone_color_converter is None:
        ckpt_converter = 'checkpoints_v2/converter'
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device='cpu')
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    return tone_color_converter

def load_ckpt(color_converter):
    ckpt_converter = get_ckpt_converter_path()
    color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

def get_reference_speaker(voice_file=None):
    openvoice_path = get_openvoice_path()
    voice_file = voice_file or 'mickey.mp3'
    print(f'Voice file: {voice_file}')
    return f'{openvoice_path}/resources/{voice_file}'


def generate_output_wav_path():
    return f'{get_output_dir()}/tmp.wav'

def get_se_and_audio(refspkr=None, tone_color_converter=None):
    tcc = tone_color_converter or get_tone_color_converter()
    ref_speaker = refspkr or get_reference_speaker('mickey.mp3')

    return se_extractor.get_se(ref_speaker, tcc, vad=True)

def get_sample_texts():
    sample_texts = {
        "EN": "Hey Aliana and Ava! Are you ready for your incredible trip to Walt Disney World?"
    }
    return sample_texts

def get_device():
    return 'cpu'

def tts(texts=None, speed=1.0):
    texts = texts or get_sample_texts()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f'{get_output_dir()}/tts_{timestamp}.wav'

    for language, text in texts.items():
        model = TTS(language=language, device=get_device())
        speaker_ids = model.hps.data.spk2id

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')

        model.tts_to_file(text, speaker_id, wav_path, speed=speed)
    return wav_path

def tone_color_convert(tts_wav_path):
    tone_color_converter = get_tone_color_converter()
    reference_speaker = get_reference_speaker()
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)
    source_se = torch.load(f'checkpoints_v2/base_speakers/ses/en-newest.pth', map_location=get_device())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_v2_en_{timestamp}.wav"
    save_path = os.path.join('outputs_v2', output_filename)
    encode_message = "@MyShell"
    tone_color_converter.convert(audio_src_path=tts_wav_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)
    return save_path

def run_demo_2(texts=None, speed=0.9):
    tts_wav_path = tts(texts, speed)
    output_file = tone_color_convert(tts_wav_path, )

def run_demo(speed=1.0):
    ckpt_base = 'checkpoints/base_speakers/EN'
    ckpt_converter = 'checkpoints/converter'
    #device="cuda:0" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    output_dir = 'outputs'

    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)
    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

    reference_speaker = 'resources/mickey.mp3'
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

    save_path = f'{output_dir}/output_en_default.wav'

    # Run the base speaker tts
    text = "Hey Aliana and Ava!!"
    src_path = f'{output_dir}/tmp.wav'
    base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)

    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
    audio_src_path=src_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=save_path,
    message=encode_message)

if __name__ == '__main__':
    run_demo_2()

