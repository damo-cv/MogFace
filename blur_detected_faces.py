from tqdm import tqdm
import traceback
import os
from fractions import Fraction
from ffprobe3 import FFProbe
import cv2
import pandas as pd
import numpy as np


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
    out = cv2.VideoWriter(
        blurred_video_file, fourcc, video_fps, image_size)
    for frame in blurred_video:
        out.write(frame)
    out.release()

def apply_blurring_on_faces(video, faces_boxes):
    blurring_kernel_size = (np.array(video[0].shape[:2])/100).astype(int)
    for frame_id, frame in enumerate(video):
        frame_boxes = faces_boxes[faces_boxes["frame_id"] == frame_id]
        if not len(frame_boxes):
            continue
        for _, xmin, ymin, w, h, _ in frame_boxes.values:
            xmax = xmin+w
            ymax = ymin+h
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            blurred_frame = cv2.blur(frame, blurring_kernel_size, 0)
            frame[ymin:ymax, xmin:xmax] = blurred_frame[ymin:ymax, xmin:xmax]
    return video

def blur_detected_faces_on_video(annotation_file):
    video_name = annotation_file.replace(".txt", ".mp4")
    video_file = "/workspace/videos/" + video_name
    blurred_video_file = "blurred_videos/" + video_name
    if os.path.exists(blurred_video_file):
        return
    faces_boxes = pd.read_csv(
        "annotations/"+annotation_file, header=None, sep=" ")
    faces_boxes.columns = ["frame_id", "x1", "y1", "w", "h", "score"]
    video = load_original_video(video_file)
    blurred_video = apply_blurring_on_faces(video, faces_boxes)
    write_video(blurred_video, video_file, blurred_video_file)

def main():
    annotations_files = os.listdir("annotations")
    for annotation_file in tqdm(annotations_files):
        blur_detected_faces_on_video(annotation_file)

if __name__ == "__main__":
    main()
