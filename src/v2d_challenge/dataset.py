import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Track1Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.metadata = {row['episode_index']: row for row in read_jsonl(self.root / 'meta/episodes_metadata.jsonl')}
        self.episodes = {row['episode_index']: row for row in read_jsonl(self.root / 'meta/episodes.jsonl')}

    def video_path(self, episode):
        metadata = self.metadata[episode]
        return self.root / self.info['video_path'].format(
            episode_chunk=episode // self.info['chunks_size'], episode_index=episode, video_key=metadata['video_key'])

    def index(self, episode):
        path = self.root / self.info['data_path'].format(
            episode_chunk=episode // self.info['chunks_size'], episode_index=episode)
        frame_index = pd.read_parquet(path)
        if len(frame_index) != self.episodes[episode]['length']:
            raise ValueError(f'Episode {episode}: inconsistent index length')
        if not np.array_equal(frame_index['frame_index'], np.arange(len(frame_index))):
            raise ValueError(f'Episode {episode}: frame IDs are not contiguous')
        return frame_index

    def frames(self, episode):
        capture = cv2.VideoCapture(str(self.video_path(episode)))
        if not capture.isOpened():
            raise ValueError(f'Episode {episode}: cannot open video')
        expected = self.episodes[episode]['length']
        decoded = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield decoded, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded += 1
            if decoded != expected:
                raise ValueError(f'Episode {episode}: decoded {decoded}, expected {expected}')
        finally:
            capture.release()
