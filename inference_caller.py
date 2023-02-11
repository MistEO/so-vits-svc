import io
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile
import sys
import os

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp(
    Path(__file__).parent / "inference/chunks_temp.json")

######
# 调用参数
# python inference_caller.py <model_path> <config_path> <hubert_path> \
#           <vocals_path> <key> <spk> <slice_db> <output> <output_format>
######

model_path = sys.argv[1]
config_path = sys.argv[2]
hubert_path = sys.argv[3]

svc_model = Svc(model_path, config_path, hubert_path)

raw_audio_path = sys.argv[4]  # 仅支持 wav
key = int(sys.argv[5])  # 音高调整，支持正负（半音）
spk = sys.argv[6]
slice_db = int(sys.argv[7])  # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
output = Path(sys.argv[8])
wav_format = sys.argv[9]  # 音频输出格式

infer_tool.format_wav(raw_audio_path)
wav_path = Path(raw_audio_path).with_suffix('.wav')
chunks = slicer.cut(wav_path, db_thresh=slice_db)
audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

audio = []
for (slice_tag, data) in audio_data:
    print(
        f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
    length = int(
        np.ceil(len(data) / audio_sr * svc_model.target_sample))
    raw_path = io.BytesIO()
    soundfile.write(raw_path, data, audio_sr, format="wav")
    raw_path.seek(0)
    if slice_tag:
        print('jump empty segment')
        _audio = np.zeros(length)
    else:
        out_audio, out_sr = svc_model.infer(spk, key, raw_path)
        _audio = out_audio.cpu().numpy()
    audio.extend(list(_audio))

res_path = output / f'{wav_path.parent.stem}_{key}key_{spk}.{wav_format}'
os.makedirs(output, exist_ok=True)
soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
