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
    torchaudio.save( tts_wav, tts_speech, target_sr)
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
