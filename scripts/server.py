import re
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Iterator
import os
import uvicorn
from omegaconf import OmegaConf
from .pipeline import InferenceExecutor
import json
import logging
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log message format
    handlers=[
        logging.StreamHandler()  # Outputs log messages to the console
    ]
)

# Get a logger instance
logger = logging.getLogger(__name__)
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load inference config
# inference_config = OmegaConf.load("configs/inference/realtime.yaml")
inference_config = OmegaConf.load("configs/inference/realtime.yaml")

print(inference_config)

inference_executors = {
#   "trump": InferenceExecutor("trump",inference_config, 1),
#   "Portrait-of-Dr.-Gachet": InferenceExecutor("Portrait-of-Dr.-Gachet",inference_config, 1),
#   "yongen": InferenceExecutor("yongen",inference_config, 1),
#   "elon": InferenceExecutor("elon",inference_config, 1),
#   "boy_play_guitar": InferenceExecutor("boy_play_guitar",inference_config, 1),
#   "girl_play_guitar2": InferenceExecutor("girl_play_guitar2",inference_config, 1),
#   "seaside4": InferenceExecutor("seaside4",inference_config, 1),
#   "seaside_girl": InferenceExecutor("seaside_girl",inference_config, 1),
   "elon": InferenceExecutor("elon",inference_config, 1),
}


def get_video_stream(file_path: str, start: int = 0, end: int = None) -> Iterator[bytes]:
    with open(file_path, 'rb') as video:
        video.seek(start)
        remaining = end - start + 1 if end else None
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            read_size = min(chunk_size, remaining) if remaining else chunk_size
            data = video.read(read_size)
            if not data:
                break
            yield data
            if remaining:
                remaining -= len(data)
                if remaining <= 0:
                    break

@app.get("/test")
async def stream_video_chat_test(request: Request):
    file_path = "data/video/yongen.mp4"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")

    range_header = request.headers.get('range')
    if range_header:
        byte1, byte2 = 0, None
        m = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if m:
            g = m.groups()
            byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])
        total_size = os.path.getsize(file_path)
        byte2 = byte2 if byte2 is not None else total_size - 1
        length = byte2 - byte1 + 1
        headers = {
            'Content-Range': f'bytes {byte1}-{byte2}/{total_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(length),
            'Content-Type': 'video/mp4',
        }
        return StreamingResponse(
            get_video_stream(file_path, byte1, byte2),
            status_code=206,
            headers=headers,
        )
    else:
        return StreamingResponse(
            get_video_stream(file_path),
            media_type="video/mp4",
        )
    
def estimate_file_size(input_text: str) -> int:
    """
    Estimate the size of files based on the input text.
    Each character in the input text corresponds to 4 KB in size.
6 =
    Args:
    input_text (str): The input text.

    Returns:
    int: The estimated size of the files in bytes.
    """
    # Define the size per character in bytes (4 KB = 4096 bytes)
    size_per_character = 4
    
    # Calculate the total size
    total_size = len(input_text) * size_per_character
    
    return total_size

@app.get("/chat_live")
async def stream_video_live(request: Request,  question: str, id: str):
    # Manually read the body as a JSON object
    # body_bytes = await request.body()
    # if not body_bytes:
    #     return {"message": "Received JSON in chat_live request", "body": body_bytes}
    # body = json.loads(body_bytes)
    input_text = question #body['question']
    logger.info(f"stream_video_live input {input_text}")
    total_size = 201270000 #estimate_file_size(input_text)


    range_header = request.headers.get('range')
    headers = {
        'Content-Range': f'bytes {0}-{total_size}/{total_size}',
        'Accept-Ranges': 'bytes',
        # 'Content-Length': str(total_size),
        'Content-Type': 'video/mp4',
    }

    return StreamingResponse(
        inference_executors['elon'].run_simple_video_inference_step(input_text),
        # media_type="video/mp4",
        # status_code=206,
        # headers=headers,
    )

@app.get("/chat_offline")
async def stream_video_offline(id: str = "elon", question: str = "this is a test"):

    input_text = question
    logger.info(f"stream_video_live input {input_text}")

    result_file_path = inference_executors[id].run_block_simple_video_inference_step(input_text)
    logger.info(f"result_file_path: {result_file_path}")

    if not os.path.exists(result_file_path):
        logger.error("Video not found")
        raise HTTPException(status_code=404, detail="Video not found")

    return StreamingResponse(
                get_video_stream(result_file_path),
                media_type="video/mp4",
            )
        


    
# The main entry point
def main():
    # Run the Uvicorn server with FastAPI app
    uvicorn.run("scripts.server:app", host="0.0.0.0", port=8000, reload=False)

# If the script is run directly, invoke the main function
if __name__ == "__main__":
    main()

# uvicorn server:app --host 0.0.0.0 --port 8000
# python -m scripts.server