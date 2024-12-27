# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np
import torchaudio.transforms as T
import torch.nn as nn

class SoftLimiter(nn.Module):
    def __init__(self, threshold_db=-1.0):
        super(SoftLimiter, self).__init__()
        self.threshold_linear = 10 ** (threshold_db / 20.0)

    def forward(self, x):
        # Apply soft limiting
        x = torch.sign(x) * (1 - torch.exp(-torch.abs(x) / self.threshold_linear)) * self.threshold_linear
        return x

def process_audio(audio_tensor, target_sr=22050, gain_factor=7):
    """
    Process audio by increasing volume and applying compression and fades.

    Args:
        audio_tensor (torch.Tensor): Input audio tensor.
        target_sr (int): Target sample rate.
        gain_factor (float): Factor to amplify the volume.

    Returns:
        torch.Tensor: Processed audio tensor.
    """
    # Normalize to -1 to 1 range
    audio_tensor = audio_tensor.float() / 32768.0

    # Apply gain
    audio_tensor = audio_tensor * gain_factor

    # Define compressor (Soft Limiter)
    compressor = SoftLimiter(threshold_db=-1.0)

    # Apply compressor
    audio_tensor = compressor(audio_tensor)

    # Apply fade in/out
    fade_in_len = 100  # in samples
    fade_out_len = 100  # in samples

    # Ensure audio_tensor has enough samples
    if audio_tensor.size(-1) < (fade_in_len + fade_out_len):
        fade_in_len = fade_out_len = audio_tensor.size(-1) // 2

    # Create fade in and fade out envelopes
    fade_in = torch.linspace(0.0, 1.0, steps=fade_in_len)
    fade_out = torch.linspace(1.0, 0.0, steps=fade_out_len)

    # Apply fade in
    audio_tensor[..., :fade_in_len] *= fade_in

    # Apply fade out
    audio_tensor[..., -fade_out_len:] *= fade_out

    # Final peak limiting to prevent clipping
    audio_tensor = torch.clamp(audio_tensor, -0.95, 0.95)

    return audio_tensor

def tts(tts_text, tts_wav="", host='0.0.0.0', port=50000, mode='zero_shot', 
        spk_id='中文女', prompt_text='Hohoho! Happy holidays to all. With lots of love. From evlevenlab.', 
        prompt_wav='/mnt/data/voice/santa.m4a', instruct_text='', target_sr=22050):
    url = f"http://{host}:{port}/inference_{mode}"
    if mode == 'sft':
        payload = {
            'tts_text': tts_text,
            'spk_id': spk_id
        }
        response = requests.request("GET", url, data=payload, stream=True)
    elif mode == 'zero_shot':
        payload = {
            'tts_text': tts_text,
            'prompt_text': prompt_text
        }
        files = [('prompt_wav', ('prompt_wav', open(prompt_wav, 'rb'), 'application/octet-stream'))]
        response = requests.request("GET", url, data=payload, files=files, stream=True)
    elif mode == 'cross_lingual':
        payload = {
            'tts_text':  tts_text,
        }
        files = [('prompt_wav', ('prompt_wav', open( prompt_wav, 'rb'), 'application/octet-stream'))]
        response = requests.request("GET", url, data=payload, files=files, stream=True)
    else:
        payload = {
            'tts_text':  tts_text,
            'spk_id':  spk_id,
            'instruct_text':  instruct_text
        }
        response = requests.request("GET", url, data=payload, stream=True)
    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    logging.info('save response to {}'.format( tts_wav))
    # tts_speech = torch.clamp(tts_speech * 2, -1.0, 1.0)  # Increase volume by 100%
    torchaudio.save(tts_wav, process_audio(tts_speech), target_sr)
    logging.info('get response')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='34.170.108.64')
    parser.add_argument('--port',
                        type=int,
                        default='50000')
    parser.add_argument('--mode',
                        default='zero_shot',
                        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct'],
                        help='request mode')
    parser.add_argument('--tts_text',
                        type=str,
                        default='你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？')
    parser.add_argument('--spk_id',
                        type=str,
                        default='中文女')
    parser.add_argument('--prompt_text',
                        type=str,
                        default='Hohoho! Happy holidays to all. With lots of love. From evlevenlab.')
    parser.add_argument('--prompt_wav',
                        type=str,
                        default='santa.m4a')
    parser.add_argument('--instruct_text',
                        type=str,
                        default='Theo \'Crimson\', is a fiery, passionate rebel leader. \
                                 Fights with fervor for justice, but struggles with impulsiveness.')
    parser.add_argument('--tts_wav',
                        type=str,
                        default='demo.wav')
    args = parser.parse_args()
    prompt_sr, target_sr = 16000, 22050
    tts(args.tts_text, args.tts_wav) # args.host, args.port, args.mode, args.spk_id, args.prompt_text, args.prompt_wav, args.instruct_text)
