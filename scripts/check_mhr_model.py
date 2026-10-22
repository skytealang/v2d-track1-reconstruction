import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    started = time.monotonic()
    path = Path('checkpoints/mhr/mhr_model.pt')
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc':
        raise ValueError('Public MHR model SHA256 mismatch')
    torch.cuda.reset_peak_memory_stats()
    model = torch.jit.load(str(path), map_location='cuda').eval()
    parameters = torch.zeros((8, 204), device='cuda')
    parameters[:, 0] = torch.arange(8, device='cuda') * 0.1
    shape = torch.zeros((8, 45), device='cuda')
    expression = torch.zeros((8, 72), device='cuda')
    with torch.inference_mode():
        vertices, skeleton = model(shape, parameters, expression, True)
    if vertices.shape != (8, 18439, 3) or not torch.isfinite(vertices).all() or not torch.isfinite(skeleton).all():
        raise ValueError('MHR output topology or values invalid')
    displacement = (vertices[1] - vertices[0]).mean(dim=0)
    if displacement.norm() <= 0:
        raise ValueError('Root translation did not affect the model')
    torch.cuda.synchronize()
    result = {'model_sha256': checksum.hexdigest(), 'vertices_shape': list(vertices.shape), 'skeleton_shape': list(skeleton.shape),
              'root_translation_parameter_step': 0.1, 'native_vertex_displacement': displacement.cpu().tolist(),
              'seconds': time.monotonic() - started, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'scope': 'Public Apache-2.0 MHR forward validation on synthetic parameters; no human video reconstruction'}
    Path('artifacts/mhr_forward_check.json').write_text(json.dumps(result, indent=2))
    np.save('artifacts/mhr_synthetic_vertices.npy', (vertices * torch.tensor([1, -1, -1], device='cuda') / 100).cpu().numpy())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
