import argparse
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()
    sequence = f'episode_{args.episode:06d}'
    root = Path('data/official/track_1')
    video_path = next(root.glob(f'videos/**/{sequence}.mp4'))
    mask_path = Path('artifacts/inputs') / f'{sequence}_masks_k0.h5'
    output = args.output or Path('artifacts/visualizations') / f'{sequence}_masks_h264.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or min(width, height, frame_count) <= 0 or args.fps <= 0:
        raise ValueError('Invalid video metadata or FPS')
    command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(args.fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4',
               '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    written = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as writer, h5py.File(mask_path, 'r') as masks:
        for frame in range(frame_count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f'cannot decode frame {frame}')
            person = masks[f'{sequence}/{frame:06d}-k0.person_mask.png'][:].astype(bool)
            object_mask = masks[f'{sequence}/{frame:06d}-k0.obj_rend_mask.png'][:].astype(bool)
            if person.shape != (height, width):
                person = cv2.resize(person.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if object_mask.shape != (height, width):
                object_mask = cv2.resize(object_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = bgr.copy()
            overlay[person] = (0, 180, 0)
            overlay[object_mask] = (0, 0, 230)
            both = person & object_mask
            overlay[both] = (0, 180, 230)
            frame_image = cv2.addWeighted(bgr, 0.62, overlay, 0.38, 0)
            contours, _ = cv2.findContours(object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_image, contours, -1, (0, 0, 255), 3)
            contours, _ = cv2.findContours(person.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_image, contours, -1, (0, 255, 0), 3)
            cv2.putText(frame_image, f'{sequence}  frame {frame:04d}/{frame_count - 1:04d}', (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_image, 'green: person   red: object   yellow: overlap', (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            writer.stdin.write(frame_image.tobytes())
            written += 1
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError(f'H.264 encoding failed: {output}')
    capture.release()
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                     '-show_entries', 'stream=codec_name,nb_frames',
                                     '-of', 'json', str(output)], text=True)
    stream = json.loads(probe)['streams'][0]
    if stream['codec_name'] != 'h264' or int(stream['nb_frames']) != frame_count:
        raise RuntimeError(f'Output codec/frame count check failed: {output}')
    print({'output': str(output), 'frames': written, 'fps': args.fps, 'size_bytes': output.stat().st_size})


if __name__ == '__main__':
    main()
