import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


JOINTS = [('l_uparm', 11), ('r_uparm', 12), ('l_lowarm', 13), ('r_lowarm', 14),
          ('l_wrist', 15), ('r_wrist', 16), ('l_upleg', 23), ('r_upleg', 24),
          ('l_lowleg', 25), ('r_lowleg', 26), ('l_foot', 27), ('r_foot', 28), ('c_head', 33)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--fit-head', action='store_true')
    parser.add_argument('--landmarks-root', type=Path, default=Path('artifacts/pose_landmarks'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/mhr_landmarks_smoke'))
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    sequence = f'episode_{args.episode:06d}'
    source = np.load(args.landmarks_root / sequence / 'landmarks.npz')
    limit = min(len(source['image']), args.max_frames) if args.max_frames else len(source['image'])
    image = source['image'][:limit]
    world = source['world'][:limit]
    image = np.concatenate([image, image[:, [7, 8]].mean(axis=1, keepdims=True)], axis=1)
    world = np.concatenate([world, world[:, [7, 8]].mean(axis=1, keepdims=True)], axis=1)
    detected = np.isfinite(image[:, 0, 0]) & np.isfinite(world[:, 0, 0])
    if detected.sum() < 2:
        raise ValueError('Insufficient human detections for fitting')
    missing_count = int((~detected).sum())
    for values in [image, world]:
        for joint in range(values.shape[1]):
            for coordinate in range(3):
                valid = np.isfinite(values[:, joint, coordinate])
                values[:, joint, coordinate] = np.interp(np.arange(limit), np.where(valid)[0], values[valid, joint, coordinate])
        values[~detected, :, 3] = 0
    model = torch.jit.load('checkpoints/mhr/mhr_model.pt', map_location='cuda').eval()
    character = model.character_torch
    names = character.skeleton.joint_names
    correspondences = JOINTS if args.fit_head else JOINTS[:-1]
    model_indices = [names.index(name) for name, _ in correspondences]
    observed_indices = [index for _, index in correspondences]
    roots = []
    intrinsics = []
    for frame in range(limit):
        with np.load(f'artifacts/moge2/{sequence}/{frame:06d}.npz') as cache:
            depth = cache['depth']
            intrinsic = cache['intrinsics_normalized']
            hip = image[frame, [23, 24], :2].mean(axis=0)
            column = int(np.clip(hip[0] * depth.shape[1], 2, depth.shape[1] - 3))
            row = int(np.clip(hip[1] * depth.shape[0], 2, depth.shape[0] - 3))
            patch = depth[row - 2:row + 3, column - 2:column + 3]
            valid = np.isfinite(patch) & (patch > 0)
            if not valid.any():
                raise ValueError(f'No predicted hip depth at frame {frame}')
            distance = np.median(patch[valid])
            roots.append([float((hip[0] - intrinsic[0, 2]) / intrinsic[0, 0] * distance),
                          float((hip[1] - intrinsic[1, 2]) / intrinsic[1, 1] * distance), float(distance)])
            intrinsics.append(intrinsic)
    targets = torch.tensor(world[:, observed_indices, :3] + np.asarray(roots)[:, None, :], device='cuda', dtype=torch.float32)
    weights = torch.tensor(np.clip(image[:, observed_indices, 3], 0, 1), device='cuda')
    pixels = torch.tensor(image[:, observed_indices, :2], device='cuda')
    intrinsics = torch.tensor(np.asarray(intrinsics), device='cuda')
    flip = torch.tensor([1, -1, -1], device='cuda')
    torch.cuda.reset_peak_memory_stats()
    fitted = []
    errors = []
    projection_errors = []
    bounds = character.parameter_limits
    hip_indices = [names.index('l_upleg'), names.index('r_upleg')]
    initial_hip_errors = []
    for start in range(0, limit, args.batch_size):
        stop = min(start + args.batch_size, limit)
        count = stop - start
        parameters = torch.zeros((count, 68), device='cuda', requires_grad=True)
        with torch.no_grad():
            if fitted:
                parameters[:, 3:] = fitted[-1][-1, 3:68]
            initial_full = torch.cat([parameters, torch.zeros((count, 181), device='cuda')], dim=1)
            initial_skeleton = character.joint_parameters_to_skeleton_state(character.model_parameters_to_joint_parameters(initial_full))
            initial_hips_native = initial_skeleton[:, hip_indices, :3].mean(dim=1)
            target_hips = torch.tensor(np.asarray(roots[start:stop]), device='cuda', dtype=torch.float32)
            parameters[:, :3] = (target_hips * flip * 100 - initial_hips_native) / 10
            initialized_hips = (initial_hips_native + parameters[:, :3] * 10) / 100 * flip
            initial_hip_errors.extend(torch.linalg.vector_norm(initialized_hips - target_hips, dim=-1).cpu().tolist())
        optimizer = torch.optim.Adam([parameters], lr=0.035)
        for step in range(args.steps):
            full = torch.cat([parameters, torch.zeros((count, 181), device='cuda')], dim=1)
            joint_parameters = character.model_parameters_to_joint_parameters(full)
            skeleton = character.joint_parameters_to_skeleton_state(joint_parameters)
            predicted = skeleton[:, model_indices, :3] / 100 * flip
            projected = predicted[:, :, :2] / predicted[:, :, 2:].clamp_min(0.1)
            projected = projected * intrinsics[start:stop, None, [0, 1], [0, 1]] + intrinsics[start:stop, None, :2, 2]
            weight = weights[start:stop]
            position_loss = ((predicted - targets[start:stop]).square().sum(-1) * weight).mean()
            reprojection_loss = ((projected - pixels[start:stop]).square().sum(-1) * weight).mean()
            selected = full[:, bounds.minmax_parameter_index]
            limit_loss = (torch.relu(bounds.minmax_min - selected).square() + torch.relu(selected - bounds.minmax_max).square()).mean()
            smoothness = (parameters[1:] - parameters[:-1]).square().mean() if count > 1 else parameters.sum() * 0
            boundary = (parameters[0] - fitted[-1][-1, :68]).square().mean() if fitted else parameters.sum() * 0
            loss = position_loss + 8 * reprojection_loss + 0.02 * limit_loss + 0.001 * parameters[:, 3:].square().mean() + 0.01 * smoothness + 0.002 * boundary
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            errors.extend(torch.linalg.vector_norm(predicted - targets[start:stop], dim=-1).mean(-1).cpu().tolist())
            projection_errors.extend(torch.linalg.vector_norm(projected - pixels[start:stop], dim=-1).mean(-1).cpu().tolist())
            fitted.append(torch.cat([parameters.detach(), torch.zeros((count, 136), device='cuda')], dim=1))
        print(json.dumps({'frames': stop, 'seconds': time.monotonic() - started, 'loss': float(loss.detach())}), flush=True)
    parameters = torch.cat(fitted)
    np.savez_compressed(args.output / 'human_params.npz', pose=parameters[:, :136].cpu().numpy(),
                        scales=np.zeros(68, np.float32), shape=np.zeros(45, np.float32))
    capture = cv2.VideoCapture(f'artifacts/inputs/{sequence}.0.color.mp4')
    for frame in sorted(set([0, min(100, limit - 1), min(300, limit - 1), min(500, limit - 1), limit - 1])):
        with torch.no_grad():
            vertices, _ = model(torch.zeros((1, 45), device='cuda'), parameters[frame:frame + 1], torch.zeros((1, 72), device='cuda'), True)
            camera_vertices = (vertices[0] / 100 * flip).cpu().numpy()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, preview = capture.read()
        if not success:
            raise ValueError('Preview decode failed')
        intrinsic = intrinsics[frame].cpu().numpy()
        projected = camera_vertices[:, :2] / camera_vertices[:, 2:]
        projected = projected * np.array([intrinsic[0, 0], intrinsic[1, 1]]) + intrinsic[:2, 2]
        for point in np.round(projected[::5] * [preview.shape[1], preview.shape[0]]).astype(np.int32):
            cv2.circle(preview, tuple(point), 1, (0, 220, 0), -1)
        cv2.imwrite(str(args.output / f'{frame:06d}.jpg'), preview)
    capture.release()
    torch.cuda.synchronize()
    report = {'frames': limit, 'seconds': time.monotonic() - started, 'steps_per_batch': args.steps,
              'episode': args.episode, 'interpolated_detection_frames': missing_count,
              'root_initialization': 'Subtract untranslated MHR hip midpoint, including the seed rotation, before assigning global translation',
              'max_initial_hip_residual_m': float(max(initial_hip_errors)),
              'joint_correspondences': correspondences,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'mean_predicted_landmark_residual_m': float(np.mean(errors)),
              'mean_reprojection_residual_normalized': float(np.mean(projection_errors)),
              'method': 'Public MHR skeleton fitting to MediaPipe 3D/2D landmarks and MoGe hip depth',
              'limitations': 'Unvalidated fallback; residuals measure agreement with noisy predictions, not GT. Fixed mean shape and scales; fingers, clothing and contact are not fitted. Batch boundaries may be discontinuous.'}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
