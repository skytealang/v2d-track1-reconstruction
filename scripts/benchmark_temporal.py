import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2d_challenge.temporal import optimize_translation


def sample(seed):
    random = np.random.default_rng(seed)
    time_axis = np.arange(240) / 30
    truth = np.stack([0.4 * np.sin(time_axis * 1.3), 0.15 * np.cos(time_axis * 2.1),
                      1.2 + 0.2 * np.sin(time_axis * 0.7)], axis=1)
    truth += 0.03 * np.sin(time_axis[:, None] * np.array([6.1, 5.3, 4.7]))
    measured = truth + random.normal(0, 0.025, truth.shape)
    confidence = np.ones(len(truth))
    confidence[70:115] = 0
    confidence[165:185] = 0
    measured[confidence == 0] = np.nan
    contacts = truth + random.normal(0, 0.012, truth.shape)
    contact_weights = np.zeros(len(truth))
    contact_weights[55:130] = 1
    contact_weights[150:200] = 1
    return truth, measured, confidence, contacts, contact_weights


def metrics(prediction, truth, confidence):
    return {'position_cm': float(np.linalg.norm(prediction - truth, axis=1).mean() * 100),
            'occluded_position_cm': float(np.linalg.norm(prediction[confidence == 0] - truth[confidence == 0], axis=1).mean() * 100),
            'acceleration_error_cm_frame2': float(np.linalg.norm(np.diff(prediction, n=2, axis=0) - np.diff(truth, n=2, axis=0), axis=1).mean() * 100)}


def main():
    started = time.monotonic()
    candidates = [0.1, 1.0, 10.0, 100.0]
    tuning = {}
    for strength in candidates:
        errors = []
        for seed in range(10):
            truth, measured, confidence, contacts, weights = sample(seed)
            prediction = optimize_translation(measured, confidence, strength, contacts, weights)
            errors.append(metrics(prediction, truth, confidence)['position_cm'])
        tuning[str(strength)] = float(np.mean(errors))
    strength = float(min(tuning, key=tuning.get))
    experiments = {'linear_fill': [], 'temporal': [], 'temporal_contact': []}
    for seed in range(100, 120):
        truth, measured, confidence, contacts, weights = sample(seed)
        visible = np.flatnonzero(confidence)
        filled = np.stack([np.interp(np.arange(len(truth)), visible, measured[visible, dimension]) for dimension in range(3)], axis=1)
        predictions = {'linear_fill': filled,
                       'temporal': optimize_translation(measured, confidence, strength),
                       'temporal_contact': optimize_translation(measured, confidence, strength, contacts, weights)}
        for name, prediction in predictions.items():
            experiments[name].append(metrics(prediction, truth, confidence))
    result = {'source': 'Synthetic analytic translation trajectories; no challenge GT or videos used',
              'limitations': 'No mesh, rotation, MHR, contact estimation or official Kaggle scoring. Contact targets are noisy simulated observations.',
              'tuning_seeds': [0, 9], 'test_seeds': [100, 119], 'smoothness': strength, 'tuning_position_cm': tuning,
              'results': {name: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} for name, rows in experiments.items()},
              'seconds': time.monotonic() - started, 'gpu_memory_mib': 0}
    Path('artifacts/temporal_ablation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
