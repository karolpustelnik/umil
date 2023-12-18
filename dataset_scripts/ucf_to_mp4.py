import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor

print("Available workers (CPU cores):", os.cpu_count())


print('Start')
def create_video_from_images(images_folder, output_video_path, frame_rate):
    # Get all image file paths
    print(f"Creating video from images in {images_folder}")
    img_paths = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    if not img_paths:
        print(f"No images found in {images_folder}")
        return

    # Read the first image to determine the video resolution
    frame = cv2.imread(img_paths[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID' for AVI format
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_video_path}")

def process_video_folder(category_path, video_folder, output_category_dir, frame_rate):
    video_path = os.path.join(category_path, video_folder)
    output_video_path = os.path.join(output_category_dir, video_folder + ".mp4")

    # Check if the output video already exists
    if not os.path.exists(output_video_path):
        if os.path.isdir(video_path):
            create_video_from_images(video_path, output_video_path, frame_rate)
    else:
        print(f"Video {output_video_path} already exists. Skipping.")

def process_all_categories(base_folder, output_dir, frame_rate=30):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
        futures = []
        for category_folder in os.listdir(base_folder):
            print(f"Processing category {category_folder}")
            if category_folder != 'Normal':
                continue
            category_path = os.path.join(base_folder, category_folder)
            if os.path.isdir(category_path):
                output_category_dir = os.path.join(output_dir, category_folder)
                os.makedirs(output_category_dir, exist_ok=True)
                for video_folder in os.listdir(category_path):
                    # Submit the task to the thread pool
                    futures.append(executor.submit(process_video_folder, category_path, video_folder, output_category_dir, frame_rate))

        # Wait for all threads to complete
        for future in futures:
            future.result()


base_folder = "/media/onebid/Nowy1/ucf_reorganized/frames" # Replace with your base folder path
output_dir = "/media/onebid/Nowy1/ucf_reorganized_mp4/frames" # Replace with your desired output directory
process_all_categories(base_folder, output_dir)
