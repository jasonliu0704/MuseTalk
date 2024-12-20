import requests
import time
from pathlib import Path

def test_process_video_endpoint():
    start_time = time.time()
    processing_times = {}
    
    # API endpoint
    url = 'http://34.27.193.153:8008/process-video'
    
    # Test video file path
    video_path = Path('/Users/proudnotice/Desktop/santa-wide.mov')
    
    if not video_path.exists():
        raise FileNotFoundError(f"Test video not found at {video_path}")
    
    # Prepare files and data for upload
    upload_start = time.time()
    files = {
        'video': ('test.mp4', open(video_path, 'rb'), 'video/mp4')
    }
    
    data = {
        'text': 'Hello, this is a test message'
    }
    processing_times['upload_prep'] = time.time() - upload_start
    
    try:
        # Send POST request
        request_start = time.time()
        response = requests.post(url, files=files, data=data)
        processing_times['server_processing'] = time.time() - request_start
        
        # Check response
        if response.status_code == 200:
            print("Success!")
            # Save response video
            save_start = time.time()
            output_path = Path('test_output.mp4')
            with open(output_path, 'wb') as f:
                f.write(response.content)
            processing_times['save_output'] = time.time() - save_start
            print(f"Saved processed video to {output_path}")
            
            # Get server-side processing time if available
            server_time = response.headers.get('X-Processing-Time')
            if server_time:
                processing_times['server_time'] = float(server_time)
                
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
        total_time = time.time() - start_time
        processing_times['total'] = total_time
        
        # Print timing information
        print("\nProcessing times:")
        print(f"Upload preparation: {processing_times.get('upload_prep', 0):.2f}s")
        print(f"Server processing: {processing_times.get('server_processing', 0):.2f}s")
        print(f"Server-side time: {processing_times.get('server_time', 0):.2f}s")
        print(f"Save output: {processing_times.get('save_output', 0):.2f}s")
        print(f"Total time: {total_time:.2f}s")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Partial timing data: {processing_times}")
    
    finally:
        # Cleanup
        files['video'][1].close()

if __name__ == "__main__":
    test_process_video_endpoint()