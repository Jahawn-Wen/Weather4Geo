# Weather4Geo


This repository packages the IJCV Stage 2 extension associated with the authors' `best` experiment. The NeurIPS implementation, dataset preparation, original prompting pipeline and pretrained-backbone setup are available in [WeatherPrompt](https://github.com/Jahawn-Wen/WeatherPrompt).

## Contents

| File | Purpose |
| --- | --- |
| `model_stage2.py` | Active X-VLM three-view model, WeatherAligner and online text prototypes |
| `train_stage2.py` | Archived training recipe, portable paths and correct source backup |
| `stage2_utils.py` | Checkpoint saving and historical EMA update |
| `evaluate.py` | Explicit checkpoint, direction and weather evaluation |
| `configs/`, `scripts/train_best.sh` | Portable resource configuration and 75.85 recipe |
| `reference/best_7585/` | Sanitized archived options, launch command and result log |


## Dependencies and external assets

1. Clone [WeatherPrompt](https://github.com/Jahawn-Wen/WeatherPrompt).
2. Follow its instructions to obtain the X-VLM code and pretrained resources. This release imports `models.xvlm.XVLMBase` from that external X-VLM tree and only `autoaugment.py`, `random_erasing.py`, `circle_loss.py` from the NeurIPS checkout. It does **not** import NeurIPS `model.py` or `project_utils.py`.
3. Provide the Swin checkpoint, BERT tokenizer/model directory, and University-1652 data. Training requires `satellite/`, `street/`, `drone/`, `google/`, each organized by the same class IDs.
4. Provide the IJCV training caption bank separately, with schema `{ "class_id": { "normal": "...", "fog": "...", ... } }`. Class IDs retain leading zeros. Use the original bank for an experiment comparison; regenerated captions are not guaranteed equivalent.
5. Use a compatible PyTorch/torchvision and X-VLM environment. 


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

Run from the IJCV release directory:

```bash
cp configs/paths.example.env configs/paths.env
# Edit every /path/to/... value, then:
source configs/paths.env
bash scripts/train_best.sh
```

The launcher materializes Swin/BERT config files in `$RUN_DIR/configs` and writes checkpoints to `$RUN_DIR/model/best_7585`. The script starts at epoch 0, runs 210 epochs (0–209), and sets the Stage 2 switch at epoch 180. It does **not** load a NeurIPS trained checkpoint. Default batch size is 8, seed 3407, Stage 2 ramp 20 epochs, and EMA coefficient 0.995.


## Evaluation

Supply an explicit IJCV checkpoint and its training options. For example, after training:

```bash
python evaluate.py \
  --checkpoint "$RUN_DIR/model/best_7585/net_209.pth" \
  --opts "$RUN_DIR/model/best_7585/opts.yaml" \
  --test-dir "$TEST_DIR" \
  --vision-config "$RUN_DIR/configs/config_swinB_384.json" \
  --text-config "$RUN_DIR/configs/config_bert.json" \
  --direction d2s --weather all \
  --device cuda:0 --output-dir "$RUN_DIR/eval_d2s"
```

Use `--direction s2d` and a separate output directory for the reverse direction. Test data must contain the corresponding `query_drone/gallery_satellite` or `query_satellite/gallery_drone` directories. SUES-200 can use the same entry point after arranging its split/height into this layout; select that dataset explicitly through `--test-dir`.




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




