from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pathlib import Path
import uuid
from pydantic import BaseModel

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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create necessary directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("results")
TEMP_DIR = Path("temp")

for dir in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir.mkdir(exist_ok=True)

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

DEFAULT_VIDEO = Path("sample_videos/santa.mp4")

@app.post("/process-video")
async def process_video(
    request: VideoRequest,
    video: UploadFile = File(...)
):
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}_input.mp4"
    
    # Process text from request body
    text = request.text
    print("input text:", text)
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        return FileResponse(
            path=input_path,
            media_type="video/mp4",
            filename=f"processed_{video.filename}"
        )
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Cleanup temp file after sending
        if input_path.exists():
            os.remove(input_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)