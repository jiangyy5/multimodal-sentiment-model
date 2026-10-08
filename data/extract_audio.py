import os
import argparse
import cv2
from tqdm import tqdm
from data.paths import resolve_dataset_dir

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip


def extract(dataset):
    dataset_root = resolve_dataset_dir(dataset)
    input_directory_path = str(dataset_root / "Raw")
    output_directory_path = str(dataset_root / "wav")
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
    for folder in tqdm(os.listdir(input_directory_path)):
        input_folder_path = os.path.join(input_directory_path, folder)
        output_folder_path = os.path.join(output_directory_path, folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        for file_name in os.listdir(input_folder_path):
            if file_name.startswith("._"):
                continue
            parts = file_name.split(".")
            if len(parts) != 2 or parts[-1] != "mp4":
                continue
            input_file_path = os.path.join(input_folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, file_name)
            if "-edited.mp4" in output_file_path:
                output_file_path = output_file_path.replace("-edited.mp4", ".mp4")
            output_file_path = output_file_path.replace(".mp4", ".wav")
            if os.path.exists(input_file_path.replace(".mp4", "-edited.mp4")):
                continue
            if os.path.exists(output_file_path):
                continue
            try:
                video = VideoFileClip(input_file_path)
                audio = video.audio
                if audio is None:
                    video.close()
                    continue
                audio.write_audiofile(
                    output_file_path,
                    fps=16000,
                    nbytes=2,
                    codec="pcm_s16le",
                    ffmpeg_params=["-ac", "1"],
                    logger=None,
                )
                audio.close()
                video.close()
            except Exception as e:
                print(input_file_path, e)
                if "-edited.mp4" in input_file_path:
                    input_file_path = input_file_path.replace("-edited.mp4", ".mp4")
                    video = VideoFileClip(input_file_path)
                    audio = video.audio
                    if audio is None:
                        video.close()
                        continue
                    audio.write_audiofile(
                        output_file_path,
                        fps=16000,
                        nbytes=2,
                        codec="pcm_s16le",
                        ffmpeg_params=["-ac", "1"],
                        logger=None,
                    )
                    audio.close()
                    video.close()


def preprocess_video_file(filename):
    cap = cv2.VideoCapture(filename)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counter = 0
    for f in range(n_frames):
        ret, frame = cap.read()
        frame_counter += 1
        if ret:
            continue
        elif frame_counter > n_frames:
            return None
        else:
            duration = (frame_counter - 1) / fps
            print(f"Fixing bad video file: {filename}")
            return duration
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sims", help="dataset name")
    args = parser.parse_args()
    if args.dataset == "mosei":
        print("Fixing MOSEI video files!")
        invalid_files = [
            "3aIQUQgawaI/12",
            "94ULum9MYX0/2",
            "mRnEJOLkhp8/24",
            "aE-X_QdDaqQ/3",
            "94ULum9MYX0/11",
            "mRnEJOLkhp8/26",
        ]
        directory_path = str(resolve_dataset_dir("mosei") / "Raw")
        for folder in os.listdir(directory_path):
            folder_path = os.path.join(directory_path, folder)
            for file_name in os.listdir(folder_path):
                fpath = os.path.join(folder_path, file_name)
                if "-edited.mp4" in fpath:
                    continue
                if os.path.exists(fpath.replace(".mp4", "-edited.mp4")):
                    continue
                if os.path.join(folder, file_name.split(".")[0]) in invalid_files:
                    continue
                duration = preprocess_video_file(fpath)
                if duration:
                    with VideoFileClip(fpath) as video:
                        new = video.subclip(0, duration)
                        new.write_videofile(
                            fpath.replace(".mp4", "-edited.mp4"),
                            verbose=False,
                            logger=None,
                        )
    extract(args.dataset)
