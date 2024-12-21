import argparse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pathlib import Path
import subprocess
import uuid
from .client import tts
from pydantic import BaseModel
from .inference import inference
import time
import logging
from typing import Optional
from fastapi import Form
from fastapi import HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class VideoRequest(BaseModel):
    text: str = "Hello, this is a default message"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create necessary directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("results")
TEMP_DIR = Path("temp")

for dir in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir.mkdir(exist_ok=True)

DEFAULT_VIDEO = Path("/mnt/data/MuseTalk/data/video/santa-wide.mov")

@app.post("/process-video")
async def process_video(
    text: str = Form(...),
    video: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    processing_times = {}

    # Check if text exceeds 280 words
    word_count = len(text.split())
    WORD_LIMIT = 280
    if word_count > WORD_LIMIT:
        raise HTTPException(status_code=400, detail=f"Text exceeds {WORD_LIMIT} words limit")
    
    
    job_id = str(uuid.uuid4())
    logger.info(f"Starting new job {job_id}")
    logger.info(f"Request text: {text}")
    # logger.info(f"Input video filename: {video.filename}")
    
    input_path = UPLOAD_DIR / f"{job_id}_input.mp4"
    output_vid_name = f"{job_id}_output.mp4"
    output_path = OUTPUT_DIR / output_vid_name
    sound_output_path = OUTPUT_DIR / f"{job_id}_sound_output.wav"
    
    try:
        # Save uploaded file
        file_save_start = time.time()
        logger.info(f"Saving uploaded file to {input_path}")
        if video is not None:
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        else:
            logger.info("No video provided, using default video")
            if not DEFAULT_VIDEO.exists():
                logger.error("Default video not found")
                raise HTTPException(status_code=500, detail="Default video not found")
            shutil.copy(DEFAULT_VIDEO, input_path)
        processing_times['file_save'] = time.time() - file_save_start
        logger.info(f"File saved successfully in {processing_times['file_save']:.2f}s")

        # Generate voice
        logger.info("Starting TTS generation")
        tts_start = time.time()
        tts(text, tts_wav=sound_output_path)
        processing_times['tts'] = time.time() - tts_start
        logger.info(f"TTS completed in {processing_times['tts']:.2f}s")
        
        # Run inference
        logger.info("Starting video inference")
        inference_start = time.time()
        inference(
            video_path=str(input_path),
            audio_path=str(sound_output_path),
            result_dir=str(OUTPUT_DIR),
            output_vid_name=str(output_vid_name)
        )
        processing_times['inference'] = time.time() - inference_start
        logger.info(f"Inference completed in {processing_times['inference']:.2f}s")
        
        total_time = time.time() - start_time
        processing_times['total'] = total_time
        
        logger.info("Processing summary:")
        logger.info(f"- File save: {processing_times['file_save']:.2f}s")
        logger.info(f"- TTS: {processing_times['tts']:.2f}s")
        logger.info(f"- Inference: {processing_times['inference']:.2f}s")
        logger.info(f"- Total time: {total_time:.2f}s")
        
        logger.info(f"Job {job_id} completed successfully")
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"{output_vid_name}",
            headers={"X-Processing-Time": str(total_time)}
        )
        
    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}")
        logger.error(f"Partial timing data: {processing_times}")
        return {"error": str(e), "processing_times": processing_times}
        
    finally:
        # Cleanup
        logger.info("Cleaning up temporary files")
        for path in [input_path, sound_output_path]:
            if path.exists():
                os.remove(path)
                logger.debug(f"Removed temporary file: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0')
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
                        default='希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav',
                        type=str,
                        default='../../../zero_shot_prompt.wav')
    parser.add_argument('--instruct_text',
                        type=str,
                        default='Theo \'Crimson\', is a fiery, passionate rebel leader. \
                                 Fights with fervor for justice, but struggles with impulsiveness.')
    parser.add_argument('--tts_wav',
                        type=str,
                        default='demo.wav')
    args = parser.parse_args()
    prompt_sr, target_sr = 16000, 22050
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)