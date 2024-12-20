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

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

DEFAULT_VIDEO = Path("data/video/santa.mp4")

@app.post("/process-video")
async def process_video(
    request: VideoRequest,
    video: UploadFile = File(...)
):
    start_time = time.time()
    processing_times = {}
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}_input.mp4"
    output_vid_name = f"{job_id}_output.mp4"
    output_path = OUTPUT_DIR / output_vid_name
    sound_output_path = OUTPUT_DIR / f"{job_id}_sound_output.wav"
    
    try:
        file_save_start = time.time()
        if video:
            # Save uploaded file
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        else:
            # Use default video
            if not DEFAULT_VIDEO.exists():
                return {"error": "Default video not found"}
            shutil.copy(DEFAULT_VIDEO, input_path)
        processing_times['file_save'] = time.time() - file_save_start

        # generate voice
        tts_start = time.time()
        tts(request.text, tts_wav=sound_output_path)
        processing_times['tts'] = time.time() - tts_start
        
        # Run inference
        inference_start = time.time()
        inference(
            video_path=str(input_path),
            audio_path=str(sound_output_path),
            result_dir=str(OUTPUT_DIR),
            output_vid_name=str(output_vid_name)
        )
        processing_times['inference'] = time.time() - inference_start
        
        total_time = time.time() - start_time
        processing_times['total'] = total_time
        
        print(f"Processing times:")
        print(f"File save: {processing_times['file_save']:.2f}s")
        print(f"TTS: {processing_times['tts']:.2f}s")
        print(f"Inference: {processing_times['inference']:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"processed_{video.filename}",
            headers={"X-Processing-Time": str(total_time)}
        )
        
    except Exception as e:
        return {"error": str(e), "processing_times": processing_times}
        
    finally:
        # Cleanup
        for path in [input_path, sound_output_path]:
            if path.exists():
                os.remove(path)

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