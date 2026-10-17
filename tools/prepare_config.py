"""Materialize portable configuration templates without loading any model."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swin-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    checkpoint = Path(args.swin_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f'Missing Swin checkpoint: {checkpoint}')
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / 'configs/config_swinB_384.json').read_text())
    config['ckpt'] = str(checkpoint)
    (output / 'config_swinB_384.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'config_bert.json').write_text((root / 'configs/config_bert.json').read_text())
    print(output)

if __name__ == '__main__':
    main()
