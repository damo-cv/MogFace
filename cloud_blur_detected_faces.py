import os
from fractions import Fraction
from ffprobe3 import FFProbe
import cv2
import pandas as pd
import numpy as np
import boto3
from tqdm import tqdm

s3 = boto3.client('s3')
ANNOTATION_PATH = "/tmp/blurring/annotations"
ORIGINAL_VIDEOS_PATH = "/tmp/blurring/videos/originals"
BLURRED_VIDEO_PATH = "/tmp/blurring/videos/blurred"

for dir in [ANNOTATION_PATH, ORIGINAL_VIDEOS_PATH, BLURRED_VIDEO_PATH]:
    if not os.path.exists(dir):
        os.makedirs(dir)


#################################
# Functions
#################################

def read_video_fps_and_duration(path: str):
    meta_data = FFProbe(path)
    fps = None
    duration = 0.
    for stream in meta_data.streams:
        if stream.is_video():
            fps = float(Fraction(stream.avg_frame_rate))
            duration = float(Fraction(stream.duration))
    return {"fps": fps, "duration": duration}


def load_original_video(video_file):
    video_reader = cv2.VideoCapture(video_file)
    is_frame_valid, image = video_reader.read()
    video = []
    while is_frame_valid:
        video.append(image)
        is_frame_valid, image = video_reader.read()
    return video

def write_video(blurred_video, video_file, blurred_video_file):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    image_size = blurred_video[0].shape[:2][::-1]
    video_fps = read_video_fps_and_duration(video_file)["fps"]
    print(blurred_video_file)
    out = cv2.VideoWriter(blurred_video_file, fourcc, video_fps, image_size)
    for frame in blurred_video:
        out.write(frame)
    out.release()

def apply_blurring_on_faces(video, faces_boxes):
    print("Blurring...")
    blurring_kernel_size = (np.array(video[0].shape[:2])/100).astype(int)
    for frame_id, frame in tqdm(enumerate(video)):
        frame_boxes = faces_boxes[faces_boxes["frame_id"] == frame_id]
        if not len(frame_boxes):
            continue
        for _, xmin, ymin, w, h, _ in frame_boxes.values:
            xmax = xmin+w
            ymax = ymin+h
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            blurred_frame = cv2.blur(frame, blurring_kernel_size, 0)
            frame[ymin:ymax, xmin:xmax] = blurred_frame[ymin:ymax, xmin:xmax]
    print("Blurring completed!")
    return video

def blur_detected_faces_on_video(local_annotation_file_path, local_video_file_path):
    video_name = local_annotation_file_path.replace(".txt", "").replace(f"{ANNOTATION_PATH}/", "")
    blurred_video_file = f"{BLURRED_VIDEO_PATH}/{video_name}"
    print(blurred_video_file)
    if os.path.exists(blurred_video_file):
        return
    faces_boxes = pd.read_csv(local_annotation_file_path, header=None, sep=" ")
    faces_boxes.columns = ["frame_id", "x1", "y1", "w", "h", "score"]
    video = load_original_video(local_video_file_path)
    blurred_video = apply_blurring_on_faces(video, faces_boxes)
    write_video(blurred_video, local_video_file_path, blurred_video_file)

def main():
    # Load annotation file
    annotation_bucket_name = "veesion-blurring-annotations"
    annotation_file_key = "be-brico-1050-couronne-330/2023-02-25_17h16m23s_to_2023-02-25_17h16m42s_camera_60_ip_192.168.1.60_port_None.mp4.txt"
    store_name = annotation_file_key.split("/")[0]
    for dir in [ANNOTATION_PATH, ORIGINAL_VIDEOS_PATH, BLURRED_VIDEO_PATH]:
        if not os.path.exists(f"{dir}/{store_name}"):
            os.makedirs(f"{dir}/{store_name}")
    local_annotation_file_path = f"{ANNOTATION_PATH}/{annotation_file_key}"
    s3.download_file(annotation_bucket_name, annotation_file_key, local_annotation_file_path)
    print(f"Annotation file {annotation_file_key} downloaded")

    # Load original video
    video_bucket_name = "veesion-test-blurring"
    video_key = annotation_file_key.replace(".txt", "")
    local_video_file_path = f"{ORIGINAL_VIDEOS_PATH}/{annotation_file_key}"
    s3.download_file(video_bucket_name, video_key, local_video_file_path)
    print(f"Original video {video_key} downloaded")

    blur_detected_faces_on_video(local_annotation_file_path, local_video_file_path)

if __name__ == "__main__":
    main()
