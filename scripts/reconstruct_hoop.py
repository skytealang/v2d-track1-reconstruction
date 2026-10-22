import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
from scipy.optimize import least_squares
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2d_challenge.dataset import Track1Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=Path('artifacts/inputs'))
    args = parser.parse_args()
    started = time.monotonic()
    sequence = 'episode_000000'
    dataset = Track1Dataset('data/official/track_1')
    if dataset.metadata[0]['object'] != 'hula_hoop':
        raise ValueError('This reconstruction prior only applies to the hula hoop episode')
    cache = np.load(f'artifacts/moge2/{sequence}/000000.npz')
    depth = cache['depth']
    intrinsics = cache['intrinsics_normalized']
    height, width = depth.shape
    with h5py.File(args.inputs / f'{sequence}_masks_k0.h5') as stream:
        predicted = stream[f'{sequence}/000000-k0.obj_rend_mask.png'][:] > 0
    mask = cv2.resize(predicted.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    skeleton = skeletonize(mask)
    valid = skeleton & cache['mask'] & np.isfinite(depth) & (depth > 0)
    rows, columns = np.where(valid)
    if len(rows) < 30:
        raise ValueError('Insufficient ring observations')
    observed_depth = depth[rows, columns]
    points = np.stack([((columns + 0.5) / width - intrinsics[0, 2]) / intrinsics[0, 0] * observed_depth,
                       ((rows + 0.5) / height - intrinsics[1, 2]) / intrinsics[1, 1] * observed_depth,
                       observed_depth], axis=1)
    centroid = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - centroid, full_matrices=False)
    first_axis, second_axis, normal = basis
    if np.linalg.det(np.column_stack([first_axis, second_axis, normal])) < 0:
        normal = -normal
    projected = (points - centroid) @ np.stack([first_axis, second_axis], axis=1)
    design = np.column_stack([2 * projected, np.ones(len(projected))])
    solution = np.linalg.lstsq(design, np.square(projected).sum(axis=1), rcond=None)[0]
    initial = np.array([solution[0], solution[1], np.sqrt(max(solution[2] + np.square(solution[:2]).sum(), 1e-8))])
    fit = least_squares(lambda parameters: np.linalg.norm(projected - parameters[:2], axis=1) - parameters[2],
                        initial, bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, np.inf]),
                        loss='soft_l1', f_scale=0.01)
    radius = float(fit.x[2])
    residual = float(np.sqrt(np.mean(np.square(fit.fun))))
    if residual > radius * 0.25:
        raise ValueError('MoGe point observations do not support a circular hoop')
    center = centroid + first_axis * fit.x[0] + second_axis * fit.x[1]
    capture = cv2.VideoCapture(str(dataset.video_path(0)))
    success, bgr = capture.read()
    capture.release()
    if not success:
        raise ValueError('Cannot read source frame')
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tube_support = predicted & (hsv[:, :, 1] > 65)
    tube_skeleton = skeletonize(tube_support)
    thickness_source = 'High-saturation RGB support inside the predicted silhouette'
    if tube_skeleton.sum() < 30:
        tube_support = predicted
        tube_skeleton = skeletonize(tube_support)
        thickness_source = 'Predicted silhouette fallback'
    pixel_radius = cv2.distanceTransform(tube_support.astype(np.uint8), cv2.DIST_L2, 5)[tube_skeleton]
    focal_pixels = (intrinsics[0, 0] * bgr.shape[1] + intrinsics[1, 1] * bgr.shape[0]) / 2
    tube_radius = float(np.clip(np.percentile(pixel_radius, 75) * np.median(observed_depth) / focal_pixels,
                                radius * 0.005, radius * 0.06))
    mesh = trimesh.creation.torus(major_radius=radius, minor_radius=tube_radius, major_sections=96, minor_sections=12)
    colored_pixels = predicted & (hsv[:, :, 1] > 40)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    albedo = np.median(rgb[colored_pixels if colored_pixels.any() else predicted], axis=0).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(np.r_[albedo, 255], (len(mesh.vertices), 1))
    output = args.inputs / f'{sequence}_object.glb'
    mesh.export(output)
    initial_pose = np.eye(4)
    initial_pose[:3, :3] = np.column_stack([first_axis, second_axis, normal])
    initial_pose[:3, 3] = center
    np.save(args.inputs / 'episode_000000_initial_object_pose.npy', initial_pose)
    angles = np.linspace(0, 2 * np.pi, 160)
    circle = center + radius * (np.cos(angles)[:, None] * first_axis + np.sin(angles)[:, None] * second_axis)
    uv = circle[:, :2] / circle[:, 2:] * np.array([intrinsics[0, 0], intrinsics[1, 1]]) + intrinsics[:2, 2]
    uv *= np.array([bgr.shape[1], bgr.shape[0]])
    cv2.polylines(bgr, [np.round(uv).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(args.inputs / 'episode_000000_mesh_reprojection.jpg'), bgr)
    result = {'source': 'Track 1 RGB, predicted SAM2 mask and predicted MoGe depth at frame 0',
              'method': 'Robust circle fit in the estimated ring plane, with tube thickness from the predicted silhouette',
              'major_radius_m': radius, 'minor_radius_m': tube_radius, 'circle_rms_m': residual,
              'thickness_source': thickness_source,
              'plane_rms_m': float(np.sqrt(np.mean(np.square((points - centroid) @ normal)))),
              'observed_points': len(points), 'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
              'watertight': bool(mesh.is_watertight), 'seconds': time.monotonic() - started, 'gpu_memory_mib': 0,
              'limitations': 'Category-specific circular shape prior; metric scale comes from monocular depth and remains uncertain. No held-out GT, Track 2 asset, human reconstruction or full trajectory used.'}
    (args.inputs / 'episode_000000_mesh_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
