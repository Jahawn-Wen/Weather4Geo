#!/usr/bin/env bash
# 75.85 archived recipe. Run from any directory; activate the xvlm environment first.
# DRY_RUN=1 bash run_ijcv_1126.sh  prints commands without training or writing files.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ijcv_root="$script_dir"
paths_file="${IJCV_PATHS:-$ijcv_root/configs/paths.env}"
if [[ -f "$paths_file" ]]; then
    source "$paths_file"
elif [[ -n "${IJCV_PATHS:-}" ]]; then
    echo "Missing IJCV_PATHS file: $paths_file" >&2; exit 2
fi
export NEURIPS_ROOT="${NEURIPS_ROOT:-$(dirname "$ijcv_root")/WeatherPrompt}"
export XVLM_ROOT="${XVLM_ROOT:-$NEURIPS_ROOT/XVLM/X-VLM-master}"
export BERT_MODEL="${BERT_MODEL:-$NEURIPS_ROOT/pretrained/bert-base-uncased}"
python_bin="${PYTHON:-python}"
run_root="${RUN_DIR:-$ijcv_root/runs/75.85}"
for helper in autoaugment.py random_erasing.py circle_loss.py; do
    [[ -f "$NEURIPS_ROOT/$helper" ]] || {
        echo "Missing WeatherPrompt helper: $NEURIPS_ROOT/$helper; configure NEURIPS_ROOT in configs/paths.env." >&2
        exit 2
    }
done
export IJCV_DIST_URL="${IJCV_DIST_URL:-tcp://127.0.0.1:12346}"
train_dir="${TRAIN_DIR:-$NEURIPS_ROOT/dataset/University-Release/train}"
captions="${TRAIN_CAPTIONS:-$ijcv_root/data/multiweather_captions_twosent.json}"
swin="${SWIN_CHECKPOINT:-$XVLM_ROOT/dataset/swin_base_patch4_window7_224_22k.pth}"
name="${EXPERIMENT_NAME:-best_7585}"
for value in "$ijcv_root" "$NEURIPS_ROOT" "$XVLM_ROOT" "$BERT_MODEL" "$run_root" "$train_dir" "$captions" "$swin"; do
    [[ "$value" == /* ]] || { echo "Use absolute resource paths: $value" >&2; exit 2; }
done
for file in "$ijcv_root/train_stage2.py" "$ijcv_root/tools/prepare_config.py" "$XVLM_ROOT/models/xvlm.py" "$captions" "$swin"; do
    [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 2; }
done
for directory in "$BERT_MODEL" "$train_dir/satellite" "$train_dir/street" "$train_dir/drone" "$train_dir/google"; do
    [[ -d "$directory" ]] || { echo "Missing directory: $directory" >&2; exit 2; }
done
prepare=("$python_bin" "$ijcv_root/tools/prepare_config.py" --swin-checkpoint "$swin" --output-dir "$run_root/configs")
train=("$python_bin" "$ijcv_root/train_stage2.py"
    --name "$name" --experiment_name "$name"
    --data_dir "$train_dir" --captions "$captions"
    --xvlm_config "$run_root/configs/config_swinB_384.json"
    --xvlm_text_config "$run_root/configs/config_bert.json"
    --views 3 --droprate 0.5 --extra_Google --share --stride 1
    --h 384 --w 384 --lr 0.005 --gpu_ids "${GPU_IDS:-1}"
    --batchsize 8 --seed 3407 --epochs 210 --num_workers "${WORKERS:-8}"
    --norm spade --reptile --qwen --iaa --focal --multi_weather
    --btnk 0 1 1 0 0 0 0 --conv_norm none --adain a
    --lambda_scene 0.10 --lambda_weather 0.03 --lambda_weather_inv 0.01
    --stage2_start_epoch 180 --stage2_ramp_epochs 20 --proto_m 0.05
    --moving_avg 0.995 "$@")
printf '75.85 training; output: %s/model/%s\n' "$run_root" "$name"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${prepare[@]}"; printf '\n'
    printf 'cd %q\n' "$run_root"
    printf '%q ' "${train[@]}"; printf '\n'
    exit 0
fi
# Refuse accidental reuse of a run directory containing checkpoints/options.
if [[ -e "$run_root/model/$name/opts.yaml" ]]; then
    echo "Existing run: $run_root/model/$name. Set a new RUN_DIR or EXPERIMENT_NAME." >&2
    exit 2
fi
mkdir -p "$run_root"
"${prepare[@]}"
cd "$run_root"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$run_root/.matplotlib}"
exec "${train[@]}"
