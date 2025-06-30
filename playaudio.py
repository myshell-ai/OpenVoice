import sys
from pydub import AudioSegment
from pydub.playback import play

def run(audio_file):
    sound = AudioSegment.from_file(audio_file)
    play(sound)

if __name__ == '__main__':
    audio_file = sys.argv[1]
    print(f'audio file: {audio_file}')
    run(audio_file)
