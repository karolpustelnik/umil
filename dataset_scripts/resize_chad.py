import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
num_cpu_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cpu_cores}")
# Function to resize a single video
def resize_video(input_path, output_path):
    target_resolution = (512, 512)

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, target_resolution)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, target_resolution)
        out.write(resized_frame)

    cap.release()
    out.release()

# Input and output directories
input_directory = '/workspace/chad'
output_directory = '/workspace/chad512'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List all video files in the input directory
video_files = [f for f in os.listdir(input_directory) if f.endswith(('.mp4', '.avi', '.mkv'))]

# Create a ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_cpu_cores) as executor:
    # Define a function to process each video file
    def process_video(video_file):
        input_path = os.path.join(input_directory, video_file)
        output_path = os.path.join(output_directory, video_file)
        resize_video(input_path, output_path)
        print(f"Video '{video_file}' resized and saved to '{output_path}'")

    # Submit video processing tasks to the ThreadPoolExecutor
    futures = [executor.submit(process_video, video_file) for video_file in video_files]

    # Wait for all submitted tasks to complete
    for future in futures:
        future.result()

print("All videos resized and saved to", output_directory)
