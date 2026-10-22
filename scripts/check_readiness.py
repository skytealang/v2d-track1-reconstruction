import json
import os
import shutil
from pathlib import Path


def main():
    home = Path.home()
    checks = {
        'track1_audit': Path('artifacts/data_audit.json').exists(),
        'docker': shutil.which('docker') is not None,
        'native_runtime': Path('.envs/cari4d/bin/python').is_file(),
        'hf_token_present': bool(os.environ.get('HF_TOKEN')) or (home / '.cache/huggingface/token').exists(),
        'kaggle_credentials_present': bool(os.environ.get('KAGGLE_API_TOKEN')) or any(
            (home / path).exists() for path in ['.kaggle/access_token', '.kaggle/kaggle.json', '.config/kaggle/credentials.json']),
        'moge_weights': Path('checkpoints/moge2/model.pt').exists(),
        'cari4d_weights': Path('checkpoints/cari4d/cari4d/2026-08-25-09-35-57/step200000.pth').exists(),
        'human_model': Path('checkpoints/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt').exists(),
        'predicted_masks_first_episode': Path('artifacts/inputs/episode_000000_masks_k0.h5').exists(),
        'reconstructed_object_first_episode': Path('artifacts/inputs/episode_000000_object.glb').exists(),
        'github_repository_configured': bool(os.environ.get('V2D_COMMIT_URL')),
    }
    result = {'checks': checks, 'full_reconstruction_ready': (checks['docker'] or checks['native_runtime']) and all(checks[key] for key in [
        'track1_audit', 'cari4d_weights', 'human_model', 'predicted_masks_first_episode', 'reconstructed_object_first_episode']),
        'note': 'Credential presence does not prove model license approval or competition entry. Never use Track 2 assets.'}
    Path('artifacts/readiness.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
