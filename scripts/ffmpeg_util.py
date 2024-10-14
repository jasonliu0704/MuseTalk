import argparse
import av
import numpy as np
import cv2
import io
# import librosa
import ffmpeg
import subprocess
from fractions import Fraction

sample_rate = 24000
duration = 1.0  # seconds
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# frequency = 440  # Hz

# combine audio and video frame into a mixture
def combine_audio_video(wav_chunk, video_frames, fps=25, save=False):
    # Generate a sequence of image frames (e.g., 30 frames for 1 second of video at 30 fps)
    frame_width, frame_height = 640, 480
    num_frames = len(video_frames)
    print(f"number of video frames: {num_frames}")

    # Create an in-memory bytes buffer
    output_buffer = io.BytesIO()
    # Open an output container using PyAV with the in-memory buffer
    output_container = av.open(output_buffer, mode='w', format='mp4')


    # Add a video stream to the container
    video_stream = output_container.add_stream('h264', rate=fps)
    video_stream.width = frame_width
    video_stream.height = frame_height
    video_stream.pix_fmt = 'yuv420p'
    # Set the time base for the video stream
    video_stream.time_base = Fraction(1, fps)
    

    # Add an audio stream to the container
    audio_stream = output_container.add_stream('aac', rate=sample_rate)
    # audio_stream.channels = 1
    audio_stream.layout = 'mono'
    # audio_stream.sample_fmt = 'fltp'
    # Set the time base for the audio stream
    audio_stream.time_base = Fraction(1, sample_rate)

    # Initialize frame PTS counters
    video_pts = 0
    audio_pts = 0

    # Write video frames to the container
    for frame_idx, img in enumerate(video_frames):
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        frame.pts = video_pts
        video_pts += 1  # Increment by 1 since time_base is 1/fps

        packets = video_stream.encode(frame)
        for packet in packets:
            output_container.mux(packet)
            
    # Flush video encoder
    packets = video_stream.encode()
    for packet in packets:
        output_container.mux(packet)

    # Calculate the number of samples per audio frame
    samples_per_frame = sample_rate // fps  # Number of audio samples per video frame

    # # Add an audio stream to the container
    # audio_stream = output_container.add_stream('aac', rate=sample_rate)
    # # audio_stream.codec_context.channel_layout = 'mono'
    # audio_stream.layout = 'mono'
    # audio_stream.codec_context.sample_rate = 44100

    # audio_frame = av.AudioFrame.from_ndarray(wav_chunk.reshape(1,-1), format='flt', layout='mono')

    # audio_frame.sample_rate = sample_rate
    # packet = audio_stream.encode(audio_frame)
    # if packet:
    #     output_container.mux(packet)

    # Ensure wav_chunk is a 1D numpy array (mono)
    audio = wav_chunk.flatten()
    # Calculate total expected audio samples
    total_video_duration = num_frames / fps  # in seconds
    expected_audio_samples = int(sample_rate * total_video_duration)
    # Check if audio length is sufficient
    if len(audio) < expected_audio_samples:
        padding_needed = expected_audio_samples - len(audio)
        print(f"Audio is shorter than expected by {padding_needed} samples. Padding audio.")
        audio = np.pad(audio, (0, padding_needed), 'constant')
    elif len(audio) > expected_audio_samples:
        print(f"Audio is longer than expected. Truncating audio.")
        audio = audio[:expected_audio_samples]

    print(f"Adjusted audio length: {len(audio)} samples")

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
        audio_frame.pts = audio_pts
        audio_pts += samples_per_frame  # Increment by the number of samples since time_base is 1/sample_rate

        packets = audio_stream.encode(audio_frame)
        for packet in packets:
            output_container.mux(packet)

    # Flush audio encoder
    packets = audio_stream.encode()
    for packet in packets:
        output_container.mux(packet)

    # Close the output container
    output_container.close()

    # Retrieve the MP4 video data from the in-memory buffer
    output_buffer.seek(0)
    mp4_data = output_buffer.read()

    # Optionally, save the MP4 data to a file to verify the output
    if save:
        with open('output_video.mp4', 'wb') as f:
            f.write(mp4_data)

    print("MP4 video has been created in memory and saved as 'output_video.mp4'.")

    return mp4_data


def wav_to_numpy(mp3_file):
    # Use FFmpeg to decode MP3 to raw PCM data
    command = [
        'ffmpeg',
        '-i', mp3_file,
        '-f', 'f32le',  # Output format: 32-bit float, little endian
        '-acodec', 'pcm_f32le',  # Audio codec
        '-'
    ]

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check for errors
    if process.returncode != 0:
        print(stderr.decode())
        raise Exception("FFmpeg error")

    # Get audio parameters using FFprobe
    import json

    probe_command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels',
        '-of', 'json',
        mp3_file
    ]

    probe_process = subprocess.Popen(probe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    probe_stdout, probe_stderr = probe_process.communicate()

    if probe_process.returncode != 0:
        print(probe_stderr.decode())
        raise Exception("FFprobe error")

    probe_info = json.loads(probe_stdout.decode())
    sample_rate = int(probe_info['streams'][0]['sample_rate'])
    channels = int(probe_info['streams'][0]['channels'])

    # Convert raw data to NumPy array
    audio_np = np.frombuffer(stdout, dtype=np.float32)

    # Reshape the array to separate channels
    if channels > 1:
        audio_np = audio_np.reshape(-1, channels)
    else:
        audio_np = audio_np.reshape(-1, 1)

    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Audio Data Shape: {audio_np.shape}")  # (n_samples, n_channels)
    return audio_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=25)

    audio_array = wav_to_numpy('../data/audio/sun.wav')

    # Mock data generation

    # Generate a sequence of image frames (e.g., 30 frames for 1 second of video at 30 fps)
    frame_width, frame_height = 640, 480
    num_frames = 30
    fps = 30
    frames = []

    for i in range(num_frames):
        # Create a blank image with a text label indicating the frame number
        img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.putText(img, f"Frame {i+1}", (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        frames.append(img)

    combine_audio_video(audio_array, frames, fps, True)