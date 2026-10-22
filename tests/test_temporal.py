import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2d_challenge.temporal import optimize_translation


def test_occluded_affine_motion_is_recovered():
    truth = np.arange(40)[:, None] * np.array([[0.1, -0.02, 0.05]]) + np.array([[1, 2, 3]])
    confidence = np.ones(40)
    confidence[8:30] = 0
    observed = truth.copy()
    observed[confidence == 0] = np.nan
    np.testing.assert_allclose(optimize_translation(observed, confidence), truth, atol=1e-8)


def test_contact_observations_can_anchor_an_invisible_clip():
    targets = np.tile([1, 2, 3], (20, 1))
    actual = optimize_translation(np.full((20, 3), np.nan), np.zeros(20),
                                  contact_targets=targets, contact_weights=np.ones(20))
    np.testing.assert_allclose(actual, targets, atol=1e-8)


def test_no_observations_fails():
    with pytest.raises(ValueError, match='anchored'):
        optimize_translation(np.full((20, 3), np.nan), np.zeros(20))


def test_invalid_visible_observation_fails():
    with pytest.raises(ValueError, match='finite'):
        optimize_translation(np.full((20, 3), np.nan), np.ones(20))


def test_missing_frames_without_regularization_fails():
    with pytest.raises(ValueError, match='smoothness'):
        optimize_translation(np.zeros((3, 3)), np.array([1, 0, 1]), smoothness=0)
