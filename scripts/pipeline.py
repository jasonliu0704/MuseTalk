import io
import os
import sys
import time
import zipfile
import argparse
from omegaconf import OmegaConf
import logging

import numpy as np

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from typing import Optional

import datetime
import os
import zipfile
from io import BytesIO
import requests
import uuid

# from ChatTTS.tools.audio import pcm_arr_to_mp3_view
# from ChatTTS.tools.logger import get_logger
import torch
# from ChatTTS.tools.normalizer.en import normalizer_en_nemo_text
# from ChatTTS.tools.normalizer.zh import normalizer_zh_tn
# from ChatTTS.tools.audio import pcm_arr_to_mp3_view

from .realtime_inference import Avatar
from pydantic import BaseModel

logger = logging.getLogger(__name__)
# Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

#chat tts
# def load_normalizer(chat: ChatTTS.Chat):
#     # try to load normalizer
#     try:
#         chat.normalizer.register("en", normalizer_en_nemo_text())
#     except ValueError as e:
#         logger.error(e)
#     except BaseException:
#         logger.warning("Package nemo_text_processing not found!")
#         logger.warning(
#             "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
#         )
#     try:
#         chat.normalizer.register("zh", normalizer_zh_tn())
#     except ValueError as e:
#         logger.error(e)
#     except BaseException:
#         logger.warning("Package WeTextProcessing not found!")
#         logger.warning(
#             "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
#         )

# def save_mp3_file(wav, index):
#     data = pcm_arr_to_mp3_view(wav)
#     mp3_filename = f"output_audio_{index}.mp3"
#     with open(mp3_filename, "wb") as f:
#         f.write(data)
#     logger.info(f"Audio saved to {mp3_filename}")


chattts_service_host = os.environ.get("CHATTTS_SERVICE_HOST", "localhost")
chattts_service_port = os.environ.get("CHATTTS_SERVICE_PORT", "8001")

CHATTTS_URL = f"http://{chattts_service_host}:{chattts_service_port}/generate_voice_chat_stream"

def chattts_infer(text:str, stream=True):
    # main infer params
    body = {
        "text": [
            text
        ],
        "stream": stream,
        "lang": None,
        "skip_refine_text": True,
        "refine_text_only": False,
        "use_decoder": True,
        "audio_seed": 12345678,
        "text_seed": 87654321,
        "do_text_normalization": True,
        "do_homophone_replacement": False,
    }

    # refine text params
    params_refine_text = {
        "prompt": "",
        "top_P": 0.7,
        "top_K": 20,
        "temperature": 0.7,
        "repetition_penalty": 1,
        "max_new_token": 384,
        "min_new_token": 0,
        "show_tqdm": True,
        "ensure_non_empty": True,
        "stream_batch": 24,
    }
    body["params_refine_text"] = params_refine_text

    # infer code params
    params_infer_code = {
        "prompt": "[speed_5]",
        "top_P": 0.1,
        "top_K": 20,
        "temperature": 0.3,
        "repetition_penalty": 1.05,
        "max_new_token": 2048,
        "min_new_token": 0,
        "show_tqdm": True,
        "ensure_non_empty": True,
        "stream_batch": True,
        "spk_emb": None,
    }
    body["params_infer_code"] = params_infer_code

    """
    Sends a POST request to the generate_voice endpoint and streams the WAV response to a file.

    Parameters:
    - api_url (str): The URL of the FastAPI endpoint.
    - payload (dict): The JSON payload to send in the POST request.
    - output_file_path (str): The path where the streamed WAV file will be saved.
    """
    try:
        # Send the POST request with streaming enabled
        with requests.post(CHATTTS_URL, json=body, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad status codes

            # Check if the response is a WAV stream
            content_type = response.headers.get('Content-Type', '')
            if 'audio/wav' not in content_type:
                print(f"Unexpected Content-Type: {content_type}")
                print("Response Text:", response.text)
                return

            # Open the output file in binary write mode
                
            # Iterate over the response in chunks
            # Initialize variables to keep track of the header and data
            header = b''
            data_size = 0

            # Create an iterator for the response content
            chunks = response.iter_content(chunk_size=262144)            
            # Read the WAV header (44 bytes)
            while len(header) < 44:
                try:
                    chunk = next(chunks)
                except StopIteration:
                    raise ValueError("Received incomplete WAV header.")

                if chunk:
                    header += chunk

            # Split the header and the initial data
            wav_header = header[:44]
            remaining_header = header[44:]

             # Write any remaining data from the header chunk
            if remaining_header:
                yield remaining_header
                data_size += len(remaining_header)
            
            # The the rest of non header data
            for chunk in chunks:
                if chunk:  # Filter out keep-alive chunks
                    yield chunk
                    data_size += len(chunk)
            print(f"total data_size: {data_size}")        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP error
    except Exception as err:
        print(f"An error occurred: {err}")          # Other errors


def chattts_infer_to_file(text:str, job_id:str):
    # main infer params
    body = {
        "text": [
            text
        ],
        "stream": True,
        "lang": None,
        "skip_refine_text": True,
        "refine_text_only": False,
        "use_decoder": True,
        "audio_seed": 12345678,
        "text_seed": 87654321,
        "do_text_normalization": True,
        "do_homophone_replacement": False,
    }

    # refine text params
    params_refine_text = {
        "prompt": "",
        "top_P": 0.7,
        "top_K": 20,
        "temperature": 0.7,
        "repetition_penalty": 1,
        "max_new_token": 384,
        "min_new_token": 0,
        "show_tqdm": True,
        "ensure_non_empty": True,
        "stream_batch": 24,
    }
    body["params_refine_text"] = params_refine_text

    # infer code params
    params_infer_code = {
        "prompt": "[speed_5]",
        "top_P": 0.1,
        "top_K": 20,
        "temperature": 0.3,
        "repetition_penalty": 1.05,
        "max_new_token": 2048,
        "min_new_token": 0,
        "show_tqdm": True,
        "ensure_non_empty": True,
        "stream_batch": True,
        "spk_emb": None,
    }
    body["params_infer_code"] = params_infer_code

    try:
        response = requests.post(CHATTTS_URL, json=body)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
            # save files for each request in a different folder
            dt = datetime.datetime.now()
            # ts = int(dt.timestamp())
            tgt = f"./output/{job_id}/"
            os.makedirs(tgt, 0o755)
            zip_ref.extractall(tgt)
            print("Extracted files into", tgt)

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")

@torch.no_grad() 
class InferenceExecutor:
    def __init__(self, avatar_id, inference_config:dict, batch_size:int,
                 source: str = "local",
    custom_path: str = "", stream: bool = True):
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        bbox_shift = inference_config[avatar_id]["bbox_shift"]
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        
        # # get avatar_id
        self.avatar = Avatar(
                avatar_id = avatar_id, 
                video_path = video_path, 
                bbox_shift = bbox_shift, 
                batch_size = batch_size,
                preparation= data_preparation)
        
        #init chattts instance
        # chat = ChatTTS.Chat()
        logger.info("Initializing ChatTTS...")
        # load_normalizer(chat)
        # is_load = False
        # if os.path.isdir(custom_path) and source == "custom":
        #     is_load = chat.load(source="custom", custom_path=custom_path)
        # else:
        #     is_load = chat.load(source=source)

        # if is_load:
        #     logger.info("Models loaded successfully.")
        # else:
        #     logger.error("Models load failed.")
        #     sys.exit(1)

    def run_block_simple_video_inference_step(self, texts:str,  
                                              spk: Optional[str] = None, source: str = "local", custom_path: str = ""):
        
        #chat tts
        logger.info("Text input: %s", str(texts))
        

        logger.info("Start inference.")
        # Generate a UUID
        uuid_str = str(uuid.uuid4())
        chattts_infer_to_file(texts, uuid_str)


        
        # yield self.avatar.streaming_inference(wave_result, 
        #                 "texts--" + str(0), 
        #                 args.fps,
        #                 args.skip_save_images)

        logger.info("Inference completed.")
        output_loc = self.avatar.inference(f"./output/{uuid_str}/0.mp3", 
                            "generated_video", 
                            30,
                            False)
        logger.info("run_block_simple_video_inference_step done!!!")

        # return generated file path
        return output_loc



    # a single step of inferencing: simple avatar + voice inference
    # need to chunk the text input if you want to stream this
    def run_simple_video_inference_step(self, texts: str, spk: Optional[str] = None, source: str = "local", custom_path: str = ""):
        # chat tts
        logger.info("Text input: %s", str(texts))

        logger.info("Start inference.")
        wavs_gen = chattts_infer(texts)
        for index, audio in enumerate(wavs_gen):
            logger.info("Inferring: {index}")
            print(f"Length of audio: {len(audio)} bytes")
            raw_audio = audio
            # load the buffer according to how load_audio work
            audio = np.frombuffer(audio, np.int16).flatten().astype(np.float32)

            # Ensure audio is a NumPy array
            # if isinstance(audio, torch.Tensor):
            #     audio = audio.cpu().numpy()

            # Convert to float32 and normalize if necessary
            # audio = np.frombuffer(audio, dtype=np.float32)
            # max_abs_value = max(abs(np.min(audio)), abs(np.max(audio)))
            # if max_abs_value == 0:
            #     max_abs_value = 1  # divide by zero error
            # audio = audio / max_abs_value

            # Check for NaN or Inf
            # if np.isnan(audio).any():
            #     print("Audio data contains NaN values.")
            #     audio = np.nan_to_num(audio, nan=0.0)
            # if np.isinf(audio).any():
            #     print("Audio data contains Inf values.")
            #     audio = np.nan_to_num(audio, posinf=1.0, neginf=-1.0)

            # Optionally, log statistics
            print(f"Audio statistics - min: {audio.min()}, max: {audio.max()}, mean: {audio.mean()}")

            # Convert normalized audio back to int16 format for saving
            # audio_int16 = (audio * 32767).astype(np.int16)

            # Save the audio chunk to a .wav file
            # import wave

            # output_path = os.path.join("inference_results", f"audio_result_{index}.wav")
            # with wave.open(output_path, 'wb') as af:
            #     af.setnchannels(1)  # Mono audio
            #     af.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
            #     af.setframerate(24000)  # Assuming a sample rate of 16kHz
            #     af.writeframes(raw_audio)

            logger.debug(f"self.avatar.streaming_inference stream i: {index}")
            yield self.avatar.streaming_inference(audio, 
                        "video--" + str(index), 
                        50,
                        True)

        logger.info("Inference completed.")
                
        # musetalk inference

        
        

if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", 
                        type=str, 
                        default="configs/inference/realtime.yaml",
    )
    parser.add_argument("--fps", 
                        type=int, 
                        default=50,
    )
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=4,
    )
    parser.add_argument("--skip_save_images",
                        action="store_true",
                        help="Whether skip saving images for better generation speed calculation",
    )

    args = parser.parse_args()

    # load inference config
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    ie = InferenceExecutor("avator_1",inference_config, args.batch_size)
    text_input = "'I will never yield': Trump delivers defiant speech at site of his attempted assassination"
    output_directory = "inference_results"
    os.makedirs(output_directory, exist_ok=True)

    # ie.run_block_simple_video_inference_step("this is a test")
    with open(os.path.join(output_directory, "final_frame_result.mp4"), "ab") as f:
            
        for index, result in enumerate(ie.run_simple_video_inference_step(text_input)):
            # Save the result to a text file for demonstration purposes
            # output_path = os.path.join(output_directory, f"frame_result_{index}.mp4")
            f.write(result)

        

    logger.info("Inference testing and saving completed.")