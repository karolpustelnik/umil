import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor

print("Available workers (CPU cores):", os.cpu_count())

def create_video_from_images(images_folder, output_video_path, frame_rate):
    print(f"Creating video from images in {images_folder}")
    img_paths = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    if not img_paths:
        print(f"No images found in {images_folder}")
        return

    frame = cv2.imread(img_paths[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_video_path}")

def process_video_folder(video_path, output_video_path, frame_rate):
        if os.path.isdir(video_path):
            create_video_from_images(video_path, output_video_path, frame_rate)

def process_selected_videos(base_folder, output_dir, selected_videos, frame_rate=30):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for rel_path in selected_videos:
            full_video_path = os.path.join(base_folder, rel_path)
            output_category_dir = os.path.join(output_dir, os.path.dirname(rel_path))
            os.makedirs(output_category_dir, exist_ok=True)

            output_video_path = os.path.join(output_category_dir, os.path.basename(rel_path) + ".mp4")
            futures.append(executor.submit(process_video_folder, full_video_path, output_video_path, frame_rate))

        for future in futures:
            future.result()


base_folder = "/media/onebid/Nowy1/ucf_reorganized/frames"  # Replace with your base folder path
output_dir = "/media/onebid/Nowy1/ucf_reorganized_mp4/frames"  # Replace with your desired output directory

selected_videos = ['Normal/Normal_Videos758_x264', "Assault/Assault011_x264", 'Normal/Normal_Videos892_x264', 'Fighting/Fighting047_x264']  # Replace with your list of video folder names

process_selected_videos(base_folder, output_dir, selected_videos)
