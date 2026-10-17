#!/usr/bin/env bash
# 75.85 paired evaluation, University-1652, ten weathers, both directions.
# Defaults: archived net_209.pth. This does not attribute the historical 75.85 row to epoch 209.
# MODEL_SOURCE=trained selects run_ijcv_1126.sh output; EPOCH=179 selects another epoch.
# DRY_RUN=1 only prints commands. No implicit newest-checkpoint selection.
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
name="${EXPERIMENT_NAME:-best_7585}"
source_kind="${MODEL_SOURCE:-archive}"
epoch="${EPOCH:-209}"
[[ "$epoch" =~ ^[0-9]{1,3}$ ]] || { echo 'EPOCH must be an integer between 0 and 999.' >&2; exit 2; }
printf -v checkpoint_name 'net_%03d.pth' "$((10#$epoch))"
case "$source_kind" in
    archive) model_dir="${ARCHIVE_DIR:-$ijcv_root/checkpoints/75.85}" ;;
    trained) model_dir="$run_root/model/$name" ;;
    *) echo 'MODEL_SOURCE must be archive or trained.' >&2; exit 2 ;;
esac
checkpoint="${CHECKPOINT:-$model_dir/$checkpoint_name}"
opts="${OPTS:-$(dirname "$checkpoint")/opts.yaml}"
test_dir="${TEST_DIR:-$NEURIPS_ROOT/dataset/University-Release/test}"
weather="${WEATHER:-all}"
direction="${DIRECTION:-both}"
case "$direction" in
    both) directions=(d2s s2d) ;;
    d2s|s2d) directions=("$direction") ;;
    *) echo 'DIRECTION must be both, d2s or s2d.' >&2; exit 2 ;;
esac
case "$weather" in all|normal|fog|rain|snow|dark|light|fog_rain|fog_snow|rain_snow|wind) ;; *) echo "Unknown WEATHER: $weather" >&2; exit 2 ;; esac
# A dated output directory avoids silently replacing results from earlier evaluations.
output_dir="${EVAL_DIR:-$run_root/evaluation/${source_kind}_$(basename "$checkpoint" .pth)_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
vision_config="${VISION_CONFIG:-$ijcv_root/configs/config_swinB_384.json}"
text_config="${TEXT_CONFIG:-$ijcv_root/configs/config_bert.json}"
for value in "$ijcv_root" "$NEURIPS_ROOT" "$XVLM_ROOT" "$BERT_MODEL" "$checkpoint" "$opts" "$test_dir" "$output_dir" "$vision_config" "$text_config"; do
    [[ "$value" == /* ]] || { echo "Use absolute resource paths: $value" >&2; exit 2; }
done
for file in "$ijcv_root/evaluate.py" "$XVLM_ROOT/models/xvlm.py" "$checkpoint" "$opts" "$vision_config" "$text_config"; do
    [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 2; }
done
[[ -d "$BERT_MODEL" ]] || { echo "Missing BERT directory: $BERT_MODEL" >&2; exit 2; }
for selected in "${directions[@]}"; do
    if [[ "$selected" == d2s ]]; then splits=(query_drone gallery_satellite); else splits=(query_satellite gallery_drone); fi
    for split in "${splits[@]}"; do
        [[ -d "$test_dir/$split" ]] || { echo "Missing test split: $test_dir/$split" >&2; exit 2; }
    done
done
printf '75.85 evaluation\nCheckpoint: %s\nOptions: %s\nData: %s\nOutput: %s\n' "$checkpoint" "$opts" "$test_dir" "$output_dir"
for selected in "${directions[@]}"; do
    evaluate=("$python_bin" "$ijcv_root/evaluate.py"
        --checkpoint "$checkpoint" --opts "$opts" --test-dir "$test_dir"
        --vision-config "$vision_config" --text-config "$text_config"
        --direction "$selected" --weather "$weather"
        --device "${DEVICE:-cuda:2}" --batch-size "${BATCH_SIZE:-128}"
        --workers "${WORKERS:-0}" --seed 3407
        --output-dir "$output_dir/$selected" "$@")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${evaluate[@]}"; printf '\n'
    else
        mkdir -p "$output_dir/$selected"
        export MPLCONFIGDIR="${MPLCONFIGDIR:-$output_dir/.matplotlib}"
        "${evaluate[@]}" 2>&1 | tee "$output_dir/$selected/console.log"
    fi
done
