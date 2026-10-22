import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def optimize_translation(observed, confidence, smoothness=10.0, contact_targets=None, contact_weights=None):
    observed = np.asarray(observed, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 3 or confidence.shape != (len(observed),):
        raise ValueError('Expected observations [T,3] and confidence [T]')
    if smoothness < 0 or not np.isfinite(smoothness):
        raise ValueError('Smoothness must be finite and nonnegative')
    if np.any(confidence < 0) or not np.isfinite(confidence).all():
        raise ValueError('Confidence must be finite and nonnegative')
    if np.any(~np.isfinite(observed[confidence > 0])):
        raise ValueError('Visible observations must be finite')
    observed = np.where(confidence[:, None] > 0, observed, 0)
    weights = confidence.copy()
    targets = confidence[:, None] * observed
    if contact_targets is not None:
        contact_targets = np.asarray(contact_targets, dtype=np.float64)
        contact_weights = np.asarray(contact_weights, dtype=np.float64)
        if contact_targets.shape != observed.shape or contact_weights.shape != confidence.shape:
            raise ValueError('Contact shapes differ from observation shapes')
        if np.any(contact_weights < 0) or not np.isfinite(contact_weights).all():
            raise ValueError('Contact weights must be finite and nonnegative')
        if not np.isfinite(contact_targets[contact_weights > 0]).all():
            raise ValueError('Active contact targets must be finite')
        contact_targets = np.where(contact_weights[:, None] > 0, contact_targets, 0)
        weights += contact_weights
        targets += contact_weights[:, None] * contact_targets
    if np.count_nonzero(weights) < min(2, len(weights)) or len(weights) == 0:
        raise ValueError('At least two distinct anchored frames are required')
    if smoothness == 0 and np.any(weights == 0):
        raise ValueError('Missing frames require positive smoothness')
    system = sparse.diags(weights, format='csc')
    if len(observed) > 2:
        acceleration = sparse.diags([np.ones(len(observed) - 2), -2 * np.ones(len(observed) - 2),
                                     np.ones(len(observed) - 2)], [0, 1, 2], shape=(len(observed) - 2, len(observed)))
        system += smoothness * acceleration.T @ acceleration
    if len(observed) <= 2 and np.any(weights == 0):
        raise ValueError('Short clips need observations at every frame')
    return np.asarray(spsolve(system, targets)).reshape(observed.shape)
