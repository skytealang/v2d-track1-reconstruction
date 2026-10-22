import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2d_challenge.dataset import Track1Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    sequence = f'episode_{args.episode:06d}'
    cache = np.load(f'artifacts/moge2/{sequence}/000000.npz')
    intrinsics = cache['intrinsics_normalized']
    with h5py.File(f'artifacts/inputs/{sequence}_masks_k0.h5') as masks:
        full_mask = masks[f'{sequence}/000000-k0.obj_rend_mask.png'][:] > 0
    depth = cache['depth']
    mask = cv2.resize(full_mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = mask & cache['mask'] & np.isfinite(depth) & (depth > 0)
    if valid.sum() < 6:
        raise ValueError('Too few object depth observations')
    rows, columns = np.where(mask)
    padding = 4
    left, right = max(0, int(columns.min()) - padding), min(mask.shape[1], int(columns.max()) + padding + 1)
    top, bottom = max(0, int(rows.min()) - padding), min(mask.shape[0], int(rows.max()) + padding + 1)
    crop_width, crop_height = right - left, bottom - top
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled_depth = depth[nearest[0], nearest[1]]
    median_depth = float(np.median(depth[valid]))
    metric_width = crop_width / depth.shape[1] / intrinsics[0, 0] * median_depth
    metric_height = crop_height / depth.shape[0] / intrinsics[1, 1] * median_depth
    thickness = float(np.clip(min(metric_width, metric_height) * 0.25, 0.015, 0.2))
    for resolution in [160, 120, 90, 68, 51, 38, 29, 22, 16]:
        scale = min(1.0, resolution / max(crop_width, crop_height))
        width, height = max(12, round(crop_width * scale)), max(12, round(crop_height * scale))
        cropped_mask = cv2.resize(mask[top:bottom, left:right].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        cropped_mask = np.pad(cropped_mask, 2)
        surface = cv2.resize(filled_depth[top:bottom, left:right], (width, height))
        surface = gaussian_filter(np.pad(surface, 2, mode='edge'), 1.0)
        volume = np.zeros((*cropped_mask.shape, 10), np.float32)
        volume[:, :, 2:8] = cropped_mask[:, :, None]
        if not cropped_mask.any():
            continue
        grid_vertices, faces, _, _ = marching_cubes(volume, level=0.5)
        pixel_rows = (grid_vertices[:, 0] - 2 + 0.5) / height * crop_height + top
        pixel_columns = (grid_vertices[:, 1] - 2 + 0.5) / width * crop_width + left
        observed_depth = map_coordinates(surface, grid_vertices[:, :2].T, order=1, mode='nearest')
        observed_depth += (grid_vertices[:, 2] - 1.5) / 6 * thickness
        vertices = np.column_stack([((pixel_columns + 0.5) / depth.shape[1] - intrinsics[0, 2]) / intrinsics[0, 0] * observed_depth,
                                    ((pixel_rows + 0.5) / depth.shape[0] - intrinsics[1, 2]) / intrinsics[1, 1] * observed_depth,
                                    observed_depth])
        center = np.median(vertices, axis=0)
        mesh = trimesh.Trimesh(vertices=vertices - center, faces=faces, process=True)
        mesh.fix_normals()
        if len(mesh.faces) > 3500:
            mesh = mesh.simplify_quadric_decimation(face_count=3500)
        if mesh.is_watertight and len(mesh.faces) <= 3500 and np.isfinite(mesh.vertices).all() and mesh.area > 0:
            break
    else:
        raise ValueError('Cannot reconstruct a watertight mesh within the face budget')
    dataset = Track1Dataset('data/official/track_1')
    capture = cv2.VideoCapture(str(dataset.video_path(args.episode)))
    success, bgr = capture.read()
    capture.release()
    if not success:
        raise ValueError('Cannot read texture image')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    color = np.median(rgb[full_mask], axis=0).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (len(mesh.vertices), 1))
    mesh.export(f'artifacts/inputs/{sequence}_object.glb')
    pose = np.eye(4)
    pose[:3, 3] = center
    np.save(f'artifacts/inputs/{sequence}_initial_object_pose.npy', pose)
    report = {'episode': args.episode, 'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
              'watertight': bool(mesh.is_watertight), 'surface_resolution': [width, height],
              'thickness_m': thickness, 'seconds': time.monotonic() - started,
              'gpu_memory_mib': 0, 'source': 'Track 1 frame 0 RGB, predicted SAM2 mask and MoGe depth',
              'method': 'Visible depth surface with silhouette-preserving extrusion and marching cubes',
              'limitations': 'Single-view 2.5D approximation. The unseen back surface and thickness are heuristic; no full object shape or metric accuracy is guaranteed.'}
    Path(f'artifacts/inputs/{sequence}_mesh_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
