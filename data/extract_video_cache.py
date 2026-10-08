import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from data.paths import resolve_dataset_dir


def _sample_indices(total_frames: int, num_frames: int):
    if total_frames <= 0:
        return []
    if num_frames <= 1:
        return [0]
    return np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64).tolist()


def _expand_bbox(x1, y1, x2, y2, width, height, margin):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * (1.0 + margin)
    bh = (y2 - y1) * (1.0 + margin)
    nx1 = int(max(0, cx - bw / 2.0))
    ny1 = int(max(0, cy - bh / 2.0))
    nx2 = int(min(width, cx + bw / 2.0))
    ny2 = int(min(height, cy + bh / 2.0))
    if nx2 <= nx1 or ny2 <= ny1:
        return (0, 0, width, height)
    return (nx1, ny1, nx2, ny2)


def _largest_face_bbox(detections, width, height):
    best = None
    best_area = -1
    for det in detections:
        bbox = det.location_data.relative_bounding_box
        x1 = int(max(0, bbox.xmin * width))
        y1 = int(max(0, bbox.ymin * height))
        x2 = int(min(width, (bbox.xmin + bbox.width) * width))
        y2 = int(min(height, (bbox.ymin + bbox.height) * height))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def _load_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return (ok, frame)


def extract_cache(
    csv_path,
    video_root,
    output_root,
    num_frames=32,
    image_size=224,
    face_margin=0.2,
    min_conf=0.5,
    split="all",
    skip_existing=True,
):
    df = pd.read_csv(csv_path)
    if "video_id" not in df.columns:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=[
                "video_id",
                "clip_id",
                "text",
                "label",
                "label_T",
                "label_A",
                "label_V",
                "polarity",
                "mode",
            ],
        )
    if "mode" in df.columns and split != "all":
        df = df[df["mode"] == split].reset_index(drop=True)
    output_root = Path(output_root)
    video_root = Path(video_root)
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=min_conf
    )
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = str(row["video_id"])
        clip_id_raw = str(row["clip_id"])
        clip_id = clip_id_raw
        if clip_id_raw.isdigit():
            candidate = video_root / video_id / f"{int(clip_id_raw):04d}.mp4"
            if candidate.exists():
                clip_id = f"{int(clip_id_raw):04d}"
        in_path = video_root / video_id / f"{clip_id}.mp4"
        out_dir = output_root / video_id
        out_path = out_dir / f"{clip_id}.pt"
        if skip_existing and out_path.exists():
            continue
        if not in_path.exists():
            print(f"[WARN] Missing video: {in_path}")
            continue
        cap = cv2.VideoCapture(str(in_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0:
            print(f"[WARN] Empty video: {in_path}")
            cap.release()
            continue
        indices = _sample_indices(total_frames, num_frames)
        frames = []
        masks = []
        bboxes = []
        prev_bbox = None
        for idx in indices:
            ok, frame = _load_frame(cap, idx)
            if not ok or frame is None:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
                masks.append(False)
                bboxes.append([-1, -1, -1, -1])
                continue
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)
            bbox = None
            if results.detections:
                bbox = _largest_face_bbox(results.detections, width, height)
            if bbox is None:
                if prev_bbox is not None:
                    bbox = prev_bbox
                else:
                    bbox = (0, 0, width, height)
            x1, y1, x2, y2 = _expand_bbox(*bbox, width, height, face_margin)
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                crop = rgb
                x1, y1, x2, y2 = (0, 0, width, height)
            crop = cv2.resize(
                crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR
            )
            frames.append(crop)
            masks.append(True)
            bboxes.append([x1, y1, x2, y2])
            prev_bbox = (x1, y1, x2, y2)
        cap.release()
        frame_tensor = (
            torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()
        )
        mask_tensor = torch.tensor(masks, dtype=torch.bool)
        bbox_tensor = torch.tensor(bboxes, dtype=torch.int32)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "frames": frame_tensor.to(torch.uint8),
            "mask": mask_tensor,
            "frame_idx": torch.tensor(indices, dtype=torch.int64),
            "bbox": bbox_tensor,
            "meta": {
                "fps": fps,
                "num_frames": total_frames,
                "size": image_size,
                "face_margin": face_margin,
                "video_id": video_id,
                "clip_id": clip_id,
            },
        }
        torch.save(payload, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="mosi", choices=["mosi", "mosei", "sims"]
    )
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--video_root", type=str)
    parser.add_argument("--output_root", type=str)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--face_margin", type=float, default=0.2)
    parser.add_argument("--min_conf", type=float, default=0.5)
    parser.add_argument(
        "--split", type=str, default="all", choices=["train", "valid", "test", "all"]
    )
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    dataset_root = None
    if not all((args.csv_path, args.video_root, args.output_root)):
        dataset_root = resolve_dataset_dir(args.dataset)
    extract_cache(
        csv_path=args.csv_path or dataset_root / "label.csv",
        video_root=args.video_root or dataset_root / "Raw",
        output_root=args.output_root or dataset_root / "video_cache" / "face32",
        num_frames=args.frames,
        image_size=args.size,
        face_margin=args.face_margin,
        min_conf=args.min_conf,
        split=args.split,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
