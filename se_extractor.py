import os
import glob
import torch
from glob import glob
from pydub import AudioSegment
from faster_whisper import WhisperModel

model_size = "medium"
# Run on GPU with FP16
model = None
def split_audio(audio_path, target_dir='processed'):
    global model
    if model is None:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)

    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    target_folder = os.path.join(target_dir, audio_name)
    
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)    

    # create directory
    os.makedirs(target_folder, exist_ok=True)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)

    # segments
    s_ind = 0
    start_time = None
    
    for k, w in enumerate(segments):
        # process with the time
        if k == 0:
            start_time = max(0, w.start)

        end_time = w.end

        # calculate confidence
        if len(w.words) > 0:
            confidence = sum([s.probability for s in w.words]) / len(w.words)
        else:
            confidence = 0.
        # clean text
        text = w.text.replace('...', '')

        # left 0.08s for each audios
        audio_seg = audio[int( start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]

        # segment file name
        fname = f"{audio_name}_seg{s_ind}.wav"

        # filter out the segment shorter than 1.5s and longer than 20s
        save = audio_seg.duration_seconds > 1.5 and \
                audio_seg.duration_seconds < 20. and \
                len(text) >= 2 and len(text) < 200 

        if save:
            output_file = os.path.join(wavs_folder, fname)
            audio_seg.export(output_file, format='wav')

        if k < len(segments) - 1:
            start_time = max(0, segments[k+1].start - 0.08)

        s_ind = s_ind + 1
    return wavs_folder


def get_se(audio_path, vc_model, target_dir='processed'):
    device = vc_model.device

    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    if os.path.isfile(se_path):
        se = torch.load(se_path).to(device)
        return se, audio_name
    if os.path.isdir(audio_path):
        wavs_folder = audio_path
    else:
        wavs_folder = split_audio(audio_path, target_dir)
    
    audio_segs = glob(f'{wavs_folder}/*.wav')
    if len(audio_segs) == 0:
        raise NotImplementedError('No audio segments found!')
    
    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name
