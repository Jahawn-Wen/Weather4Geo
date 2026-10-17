"""Explicit-checkpoint retrieval evaluation for the archived three-view model.

Matches its image-only eval behavior; does not activate the training-only aligner.
Weather augmentation comes from test_iaa_caption_ijcv_all.py. See reproducibility notes.
"""
import argparse
import json
import random
from pathlib import Path

WEATHERS = ['normal', 'fog', 'rain', 'snow', 'dark', 'light', 'fog_rain', 'fog_snow', 'rain_snow', 'wind']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--opts', required=True, help='Training opts.yaml')
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--vision-config', required=True)
    parser.add_argument('--text-config', required=True)
    parser.add_argument('--direction', choices=['d2s', 's2d'], required=True)
    parser.add_argument('--weather', choices=['all'] + WEATHERS, default='all')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import imgaug as ia
    import yaml
    from model_stage2 import three_view_net
    from runtime_paths import bert_model
    from weather_transforms import build_iaa_transform
    from retrieval_metrics import evaluate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ia.seed(args.seed)
    device = torch.device(args.device)
    config = yaml.safe_load(Path(args.opts).read_text())
    vision_config = str(Path(args.vision_config).resolve())
    text_config = str(Path(args.text_config).resolve())
    xcfg = json.loads(Path(vision_config).read_text())
    xcfg.update(use_swin=True, use_clip_vit=False, vision_config=vision_config,
                image_res=config['h'], patch_size=32, use_roberta=False,
                text_config=text_config, text_encoder=bert_model(), embed_dim=256,
                temp=0.07, max_tokens=256, use_mlm_loss=True, use_bbox_loss=True,
                load_vision_pretrained=False)
    model = three_view_net(config['nclasses'], config['droprate'],
                           config_xvlm=xcfg, num_weathers=10,
                           proto_m=config.get('proto_m', 0.05))
    # Fail on wrong versions/checkpoints rather than silently dropping parameter keys.
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.classifier.classifier = nn.Sequential()
    model.eval().to(device)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    satellite_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=3), normalize])
    direction = ('query_drone', 'gallery_satellite') if args.direction == 'd2s' else ('query_satellite', 'gallery_drone')
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def extract(split, weather):
        drone = 'drone' in split
        ds = datasets.ImageFolder(str(Path(args.test_dir) / split), transform=None)
        augment = build_iaa_transform(weather, config['h'], config['w'])
        # ImageFolder labels are split-local; use numeric directory names for retrieval IDs.
        labels = np.array([int(Path(path).parent.name) for path, _ in ds.samples])
        if drone:
            ds.transform = lambda image: normalize(augment(image=np.asarray(image)))
        else:
            ds.transform = satellite_transform
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
        chunks = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                values = []
                for tensor in (images, torch.flip(images, dims=[3])):
                    if drone:
                        _, _, feat = model(None, None, tensor)
                    else:
                        feat, _, _ = model(tensor, None, None)
                    values.append(feat)
                chunks.append(F.normalize(values[0] + values[1], dim=1).cpu())
        return torch.cat(chunks), labels

    results = []
    for weather in (WEATHERS if args.weather == 'all' else [args.weather]):
        # Seed each condition independently; record this policy with each result.
        random.seed(args.seed); np.random.seed(args.seed); ia.seed(args.seed)
        qf, ql = extract(direction[0], weather)
        gf, gl = extract(direction[1], weather)
        qf, gf = qf.to(device), gf.to(device)
        cmc, ap, invalid = torch.zeros(len(gl)), 0.0, 0
        for feature, label in zip(qf, ql):
            sample_ap, sample_cmc = evaluate(feature, label, gf, gl)
            if sample_cmc[0] == -1:
                invalid += 1
                continue
            cmc += sample_cmc
            ap += sample_ap
        # Same denominator as NeurIPS evaluator, including queries with no valid match.
        cmc /= len(ql)
        rank = lambda k: float(cmc[min(k-1, len(gl)-1)] * 100)
        record = dict(weather=weather, direction=args.direction, queries=len(ql), gallery=len(gl),
                      unmatched_queries=invalid, R1=rank(1), R5=rank(5), R10=rank(10),
                      legacy_top1_percent=float(cmc[min(round(len(gl)*0.01), len(gl)-1)]*100),
                      mAP=ap/len(ql)*100, checkpoint=str(Path(args.checkpoint).resolve()),
                      seed=args.seed, workers=args.workers, inference='historical_image_only')
        results.append(record)
        (output_dir / f'{args.direction}_{weather}.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    (output_dir / f'{args.direction}_summary.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
