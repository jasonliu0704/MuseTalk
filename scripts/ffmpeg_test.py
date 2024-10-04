import av
import numpy as np
import cv2
import io

# Mock data generation

# Generate a sequence of image frames (e.g., 30 frames for 1 second of video at 30 fps)
frame_width, frame_height = 640, 480
num_frames = 30
frames = []

for i in range(num_frames):
    # Create a blank image with a text label indicating the frame number
    img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.putText(img, f"Frame {i+1}", (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    frames.append(img)

# Generate audio data (e.g., a 1-second sine wave at 440Hz)
sample_rate = 44100
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
frequency = 440  # Hz
audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

# Combine image frames and audio into an MP4 video in memory

# Create an in-memory bytes buffer
output_buffer = io.BytesIO()

# Open an output container using PyAV with the in-memory buffer
output_container = av.open(output_buffer, mode='w', format='mp4')

# Add a video stream to the container
video_stream = output_container.add_stream('h264', rate=30)  # 30 fps
video_stream.width = frame_width
video_stream.height = frame_height
video_stream.pix_fmt = 'yuv420p'

# Add an audio stream to the container
audio_stream = output_container.add_stream('aac', rate=sample_rate)
# audio_stream.codec_context.channel_layout = 'mono'
audio_stream.layout = 'mono'
audio_stream.codec_context.sample_rate = 44100
# audio_stream.sample_fmt = 'fltp'

# Write video frames to the container
for img in frames:
    frame = av.VideoFrame.from_ndarray(img, format='bgr24')
    packet = video_stream.encode(frame)
    if packet:
        output_container.mux(packet)

# Flush video stream
packet = video_stream.encode()
if packet:
    output_container.mux(packet)

# Calculate the number of samples per audio frame
samples_per_frame = sample_rate // 30  # Assuming 30 fps

# Write audio frames to the container
for i in range(0, len(audio), samples_per_frame):
    samples = audio[i:i + samples_per_frame]

    # Ensure the last frame has the correct number of samples
    if len(samples) < samples_per_frame:
        samples = np.pad(samples, (0, samples_per_frame - len(samples)), 'constant')
    # Reshape samples to have shape (channels, samples)
    samples_reshaped = samples.reshape(1, -1)
    audio_frame = av.AudioFrame.from_ndarray(samples_reshaped, format='flt', layout='mono')

    audio_frame.sample_rate = sample_rate
    packet = audio_stream.encode(audio_frame)
    if packet:
        output_container.mux(packet)

# Flush audio stream
packet = audio_stream.encode()
if packet:
    output_container.mux(packet)

# Close the output container
output_container.close()

# Retrieve the MP4 video data from the in-memory buffer
output_buffer.seek(0)
mp4_data = output_buffer.read()

# Optionally, save the MP4 data to a file to verify the output
with open('output_video.mp4', 'wb') as f:
    f.write(mp4_data)

print("MP4 video has been created in memory and saved as 'output_video.mp4'.")
