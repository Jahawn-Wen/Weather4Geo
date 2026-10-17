# Weather4Geo


This repository packages the Weather4Geo Stage 2 extension associated with our archived best experiment. The NeurIPS implementation, dataset preparation, original prompting pipeline and pretrained-backbone setup are available in [WeatherPrompt](https://github.com/Jahawn-Wen/WeatherPrompt).

## Contents

| File | Purpose |
| --- | --- |
| `model_stage2.py` | Active X-VLM three-view model, WeatherAligner and online text prototypes |
| `train_stage2.py` | Archived training recipe, portable paths and correct source backup |
| `stage2_utils.py` | Checkpoint saving and historical EMA update |
| `evaluate.py` | Explicit checkpoint, direction and weather evaluation |
| `run_ijcv_1126.sh`, `run_test.sh` | Training and evaluation launchers |
| `configs/`, `scripts/train_best.sh` | Portable resource configuration and compatibility training entry |
| `reference/best/` | Sanitized archived options, launch command and result log |


## Dependencies and external assets

1. Clone [WeatherPrompt](https://github.com/Jahawn-Wen/WeatherPrompt).
2. Follow its instructions to obtain the X-VLM code and pretrained resources. This release imports `models.xvlm.XVLMBase` from that external X-VLM tree and only `autoaugment.py`, `random_erasing.py`, `circle_loss.py` from the NeurIPS checkout. It does **not** import NeurIPS `model.py` or `project_utils.py`.
3. Provide the Swin checkpoint, BERT tokenizer/model directory, and University-1652 data. Training requires `satellite/`, `street/`, `drone/`, `google/`, each organized by the same class IDs.
4. Provide the IJCV training caption bank separately, with schema `{ "class_id": { "normal": "...", "fog": "...", ... } }`. Class IDs retain leading zeros. Use the original bank for an experiment comparison; regenerated captions are not guaranteed equivalent.
5. Use a compatible PyTorch/torchvision and X-VLM environment. Install the WeatherPrompt / X-VLM dependencies first; this repository's `requirements.txt` lists supplementary requirements. The inspected X-VLM code uses Transformers 4.12.5, while optional Qwen caption generation requires a separate modern Qwen-compatible environment.


Organize `dataset` folder as follows:

```
|-- dataset/
|    |-- University-Release/
|        |-- test/
|            |-- query_drone/
|            |-- query_satellite/
|            |-- ...
|        |-- train/
|            |-- drone/
|            |-- satellite/
|            |-- ...
|    |-- SUES/
|        |-- Training/
|            |-- 150/
|            |-- 200/
|            |-- ...
|        |-- Testing/
|            |-- 150/
|            |-- 200/
|            |-- ...
|    |-- multiweather_twosent.json
|    |-- multiweather_captions_twosent_gallery_drone.json
|    |-- multiweather_captions_twosent_query_drone.json
```


## Models and Weights
*  The **Models** and **Weights** are released.
* Download The Trained Model Weights and Description:[Baidu Yun](https://pan.baidu.com/s/13j18tJ1HSbpW2GUC_O_fGA?pwd=dhyy)[dhyy] and [OneDrive](https://1drv.ms/f/c/5ebfdc24555e89f8/IgAswlJ9EbLRQrcELrIF_U_lAf1sKo4Kg6a11vflMjl2eTM?e=duwleM)).

Organize `XVLM` folder as follows:

```
|-- XVLM/
|    |-- X-VLM-master/
|        |-- accelerators/
|        |-- configs/
|        |-- ...
|        |-- Captioning_pretrain.py
|        |-- ...
|    |-- 4m_base_model_state_step_199999.th
|    |-- 16m_base_model_state_step_199999.th
|-- image_folder.py/
|-- ...
```



## Usage
### Install Requirements

We use single A6000 48G GPU for training and evaluation.

Create conda environment.

```
conda create -n weatherprompt python=3.9
conda activate weatherprompt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements.txt
```

### Datasets Prepare
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

Download [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark).

## Training

Run from the Weather4Geo release directory:

```bash
cp configs/paths.example.env configs/paths.env
# Edit the absolute resource paths and Python/GPU settings in configs/paths.env.
# The launchers automatically load this file.
DRY_RUN=1 bash run_ijcv_1126.sh
bash run_ijcv_1126.sh
```

By default, `RUN_DIR` is `runs/75.85` inside this repository; set it in your configuration to choose another location. The launcher materializes Swin/BERT config files in `$RUN_DIR/configs` and writes checkpoints to `$RUN_DIR/model/best`. The script starts at epoch 0, runs 210 epochs (0–209), and sets the Stage 2 switch at epoch 180. It does **not** load a NeurIPS trained checkpoint. Default batch size is 8, seed 3407, Stage 2 ramp 20 epochs, and EMA coefficient 0.995.


## Evaluation

Configure `ARCHIVE_DIR` in `configs/paths.env` to contain the IJCV checkpoint files and their `opts.yaml`. Then run:

```bash
# Inspect paths and commands without running evaluation.
DRY_RUN=1 bash run_test.sh

# Evaluate archived net_209.pth: University-1652, ten weather conditions, both directions.
bash run_test.sh

# Evaluate the checkpoint produced by the training launcher instead.
MODEL_SOURCE=trained bash run_test.sh

# Optional: select a different epoch, GPU, direction or weather.
EPOCH=199 DEVICE=cuda:0 DIRECTION=d2s WEATHER=fog bash run_test.sh
```

`MODEL_SOURCE=archive` is the default. The selected checkpoint is loaded strictly; no implicit latest-checkpoint lookup is used. `net_209.pth` is the default selection, not a verified attribution of the historical 75.85 result row. Evaluation follows the archived image-only `.eval()` path. This code upload has not rerun the complete training or benchmark.

Test data must contain `query_drone/gallery_satellite` for d2s or `query_satellite/gallery_drone` for s2d. SUES-200 can use the same entry point after arranging its selected split/height into this layout; set `TEST_DIR` accordingly. Results and console logs are saved under `$RUN_DIR/evaluation/`, separated by checkpoint, timestamp and direction.






## Reference

```bibtex
@article{wen2026weatherprompt,
  title={Weatherprompt: Multi-modality representation learning for all-weather drone visual geo-localization},
  author={Wen, Jiahao and Yu, Hang and Zheng, Zhedong},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={32974--32994},
  year={2026}
}
```

## ✨ Acknowledgement
- Our code is based on [XVLM](https://github.com/zengyan-97/X-VLM)
- [Qwen2.5-VL-32B](https://qwen.ai/research): Thanks a lot for the foundamental efforts!




