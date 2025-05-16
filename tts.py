import argparse
import json
import logging
import os
import shutil
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

import nltk   

def generate_audio(reference_speaker, phrases, language, tone_color_converter, device):

    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

    # texts = {
    #     # 'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    #     'EN': text,
    #     # 'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    #     # 'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    #     # 'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    #     # 'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    #     # 'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
    # }


    src_path = f'/temp/tmp.wav'

    # Speed is adjustable
    speed = 1.0

    for i,  phrase in enumerate(phrases):

        if phrase['type'] == "phrase":
            # text = {}

            # text[language] = phrase["content"]

            model = TTS(language=language, device=device)
            speaker_ids = model.hps.data.spk2id

            print(f"speaker_ids {len(speaker_ids)}")
            
            for speaker_key in speaker_ids.keys():
                speaker_id = speaker_ids[speaker_key]
                speaker_key = speaker_key.lower().replace('_', '-')
                
                source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
                model.tts_to_file(phrase["translated"], speaker_id, src_path, speed=speed)
                save_path = f'/temp/phrases/{i}_out.wav'

                # Run the tone color converter
                encode_message = "@MyShell"
                tone_color_converter.convert(
                    audio_src_path=src_path, 
                    src_se=source_se, 
                    tgt_se=target_se, 
                    output_path=save_path,
                    message=encode_message
                )

                phrases[i]["file_path"] = save_path
    return phrases
            




def options():
    parser = argparse.ArgumentParser(description='Inference code to clone and drive text to speech')
    parser.add_argument('--reference_speaker', '-s', type=str, required=True)
    parser.add_argument('--language', '-l', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = options()
    print(f"args {args}")

    with open("/temp/text.json", "r") as f:
        phrases = json.load(f)

    texts = {}

    for phrase in phrases:
        if phrase['type'] == "phrase":
            texts[args.language] = phrase["content"]

    print(f"texts {texts}")

    nltk.download('averaged_perceptron_tagger_eng')    
    
    ckpt_converter = 'checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    out_folder = "/temp/phrases/"
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    os.mkdir(out_folder)

    phrases = generate_audio(args.reference_speaker, phrases, args.language, tone_color_converter, device)


    print(f"Phrases {phrases}")


    with open("/temp/text.json", "w") as f:
        json.dump(phrases, f)
            
if __name__ == "__main__":
    main()
