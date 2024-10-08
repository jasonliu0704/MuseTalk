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
inference_config = OmegaConf.load("configs/inference/realtime.yaml")
print(inference_config)

inference_executor = InferenceExecutor("avator_1",inference_config, 1)


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
    
@app.get("/chat_live")
async def stream_video_live(request: Request):
    # Manually read the body as a JSON object
    body_bytes = await request.body()
    if not body_bytes:
        return {"message": "Received JSON in chat_live request", "body": body_bytes}
    body = json.loads(body_bytes)
    input_text = body['question']
    logger.info(f"stream_video_live input {input_text}")

    return StreamingResponse(
        inference_executor.run_simple_video_inference_step(input_text),
        media_type="video/mp4",
    )

@app.get("/chat_offline")
async def stream_video_offline(request: Request):
    # Manually read the body as a JSON object
    body_bytes = await request.body()
    if not body_bytes:
        return {"message": "Received JSON in chat_live request", "body": body_bytes}
    body = json.loads(body_bytes)
    input_text = body['question']
    logger.info(f"stream_video_live input {input_text}")

    result_file_path = inference_executor.run_block_simple_video_inference_step(input_text)
    logger.info(f"result_file_path: {result_file_path}")
    
    if not os.path.exists(result_file_path):
        logger.error("Video not found")
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
        total_size = os.path.getsize(result_file_path)
        byte2 = byte2 if byte2 is not None else total_size - 1
        length = byte2 - byte1 + 1
        headers = {
            'Content-Range': f'bytes {byte1}-{byte2}/{total_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(length),
            'Content-Type': 'video/mp4',
        }
        return StreamingResponse(
            get_video_stream(result_file_path, byte1, byte2),
            status_code=206,
            headers=headers,
        )
    else:
        return StreamingResponse(
            get_video_stream(result_file_path),
            media_type="video/mp4",
        )


    
# The main entry point
def main():
    # Run the Uvicorn server with FastAPI app
    uvicorn.run("scripts.server:app", host="0.0.0.0", port=8000, reload=True)

# If the script is run directly, invoke the main function
if __name__ == "__main__":
    main()

# uvicorn server:app --host 0.0.0.0 --port 8000
# python -m scripts.server