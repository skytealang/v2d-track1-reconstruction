import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def main():
    root = Path('artifacts/seg_large')
    review = root / 'review'
    review.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((root / 'masks').glob('*_masks_report.json')):
        report = json.loads(path.read_text())
        episode = report['episode']
        sequence = f'episode_{episode:06d}'
        count = report['expected_frames']
        assert report['complete_video'] and report['frames'] == count
        prompts = json.loads((root / 'prompts' / f'{sequence}.json').read_text())['prompts']
        selected = sorted(set([0, (count - 1) // 4, (count - 1) // 2,
                               3 * (count - 1) // 4, count - 1] +
                              [prompt['frame_index'] for prompt in prompts]))
        capture = cv2.VideoCapture(str(Path('data/official/track_1/videos/chunk-000/observation.images.exo_camera') / f'{sequence}.mp4'))
        tiles = []
        with h5py.File(root / 'masks' / f'{sequence}_masks_k0.h5') as masks:
            for frame in range(count):
                for role in ['person_mask', 'obj_rend_mask']:
                    key = f'{sequence}/{frame:06d}-k0.{role}.png'
                    assert key in masks and masks[key].shape == (1152, 1536)
            for frame in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                success, image = capture.read()
                assert success
                for role, color in [('person_mask', [0, 200, 0]), ('obj_rend_mask', [0, 0, 255])]:
                    mask = masks[f'{sequence}/{frame:06d}-k0.{role}.png'][:] > 0
                    image[mask] = (0.5 * image[mask] + 0.5 * np.array(color)).astype(np.uint8)
                tile = cv2.resize(image, (512, 384))
                cv2.putText(tile, f'{episode:02d} frame {frame}', (8, 25), 0, 0.7, (0, 0, 255), 2)
                tiles.append(tile)
        capture.release()
        cv2.imwrite(str(review / f'{sequence}.jpg'), cv2.hconcat(tiles))
        areas = report['areas']
        assert len(areas) == count and {row['frame'] for row in areas} == set(range(count))
        object_area = np.array([row['object_pixels'] for row in areas])
        person_area = np.array([row['human_pixels'] for row in areas])
        rows.append({'episode': episode, 'frames': count, 'seconds': report['seconds'],
                     'peak_allocated_mib': report['peak_allocated_mib'],
                     'empty_object_frames': int((object_area == 0).sum()),
                     'empty_human_frames': int((person_area == 0).sum()),
                     'prompt_frames': sorted({prompt['frame_index'] for prompt in prompts}),
                     'object_area_range': [int(object_area.min()), int(object_area.max())],
                     'review': str(review / f'{sequence}.jpg'), 'visually_approved': False})
    payload = {'episodes': rows, 'complete_episodes': len(rows),
               'note': 'Structural checks and sampled RGB overlays; no GT IoU or full video approval.'}
    (root / 'review_manifest.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps({'complete_episodes': len(rows), 'frames': sum(row['frames'] for row in rows)}))


if __name__ == '__main__':
    main()
