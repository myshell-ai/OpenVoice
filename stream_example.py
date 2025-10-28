from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from time import perf_counter
import pyaudio
import numpy as np
import queue
import select
import sys

converter = 'checkpoints_v2/converter'
device = 'cpu'
base_speaker = 'me.wav'
reference_speaker = 'ref.wav'

tone_color_converter = ToneColorConverter(f'{converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{converter}/checkpoint.pth')
tone_color_converter.watermark_model = None
source_se, _ = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)
target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

qout = queue.Queue(maxsize=100)
buf = queue.Queue(maxsize=100)
mul = 6
silence = np.zeros((mul * 1024, 1), dtype=np.float32).tobytes()
flag = True


def stream_callback(in_data, frame_count, time_info, status_flags):
    # print("!", qout.qsize())
    try:
        data = qout.get_nowait()
    except queue.Empty:
        print("Silence")
        data = silence
    return data, pyaudio.paContinue


def on_input(in_data, frame_count, time_info, status_flags):
    buf.put_nowait(in_data)
    # print(".", buf.qsize())
    return None, pyaudio.paContinue


p = pyaudio.PyAudio()

in_index = next(i for i in range(p.get_device_count())
                if "MacBook Air" in p.get_device_info_by_index(i)['name']
                and p.get_device_info_by_index(i)['maxInputChannels'] > 0)

bh_index = next(i for i in range(p.get_device_count())
                if "BlackHole" in p.get_device_info_by_index(i)['name']
                and p.get_device_info_by_index(i)['maxOutputChannels'] > 0)

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                output_device_index=bh_index,
                rate=24_000,
                output=True,
                frames_per_buffer=1024 * mul,
                stream_callback=stream_callback
                )

stream.start_stream()

stream_in = p.open(format=pyaudio.paFloat32,
                   input_device_index=in_index,
                   channels=1,
                   rate=24_000,
                   input=True, frames_per_buffer=mul * 1024,
                   stream_callback=on_input
                   )

stream_in.start_stream()

print("Streaming...")
# period = mul * 1024 / 24_000
space = np.array([0.01] * (mul * 1024), dtype=np.float32)


def adaptive_thr_rms(rms, hop_ms, tail_sec=0.8, base=1e-4, mult=3.0):
    hop = hop_ms / 1000.0
    tail_frames = max(1, int(tail_sec / hop))
    tail = rms[-tail_frames:] if rms.size >= tail_frames else rms
    return max(base, float(np.median(tail)) * mult)


def frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(x) - frame) // hop)
    if n <= 0:
        return np.empty(0, dtype=np.float32)
    pad = (n * hop + frame) - len(x)
    if pad > 0:
        x = np.pad(x, (0, pad), mode='constant')
    shape = (n, frame)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))


def last_silence_before_end(x: np.ndarray, sr=16000,
                            win_ms=25.0, hop_ms=10.0,
                            thr_rms=None, min_pause_ms=150.0):
    x = np.asarray(x, dtype=np.float32)
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    rms = frame_rms(x, win, hop)
    if rms.size == 0:
        return None
    if thr_rms is None:
        thr_rms = adaptive_thr_rms(rms, hop_ms, tail_sec=0.8, base=1e-3, mult=3.5)
    sil = rms <= thr_rms
    need = max(1, int(np.ceil(min_pause_ms / hop_ms)))

    run = 0
    end_win = None
    start_win = None
    for i in range(rms.size - 1, -1, -1):
        if sil[i]:
            run += 1
            if run == need:
                end_win = i + 1
                start_win = i - need + 1
        else:
            if run >= need:
                break
            run = 0
    if start_win is None:
        return None

    # start_sample = start_win * hop
    end_sample = min(len(x), int(end_win * hop + win))
    return end_sample


try:
    audio = None
    buf_in = np.array([])
    buf_out = np.array([])
    while stream.is_active():
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            line = sys.stdin.readline()
            flag = not flag

        now = perf_counter()
        audio = buf.get()
        audio = np.frombuffer(audio, dtype=np.float32)

        if flag:
            audio = np.concatenate((buf_in, audio))
            end = last_silence_before_end(audio, sr=24_000,
                                          win_ms=25, hop_ms=10,
                                          thr_rms=None, min_pause_ms=30)

            cut = end if end is not None else len(audio)
            buf_in = audio[end:]
            audio = audio[0:end]

            if audio.size > 400 and end is not None:
                audio = tone_color_converter.convert(
                    audio_src_path=audio,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=None,
                    message="")

                # audio[0] = 1.0 # debug
                # audio[-1] = 1.0

                audio = np.concatenate((buf_out, audio))
                if audio.size >= mul * 1024:
                    buf_out = audio[mul * 1024:]
                    audio = audio[:mul * 1024]
                else:
                    buf_out = audio
                    audio = space
            else:
                if len(buf_out) > 0:
                    audio = np.concatenate((buf_out, space))
                    buf_out = audio[mul * 1024:]
                    audio = audio[:mul * 1024]
                else:
                    audio = None

        if audio is not None:
            audio = audio.astype(np.float32, copy=False)
            # audio = np.clip(audio * 2, -1.0, 1.0)
            qout.put_nowait(audio.tobytes())

except KeyboardInterrupt:
    print("Exit")
    stream_in.stop_stream()
    stream_in.close()
    stream.stop_stream()
    stream.close()
    p.terminate()
