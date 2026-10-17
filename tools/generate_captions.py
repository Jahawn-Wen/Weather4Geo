#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser(description='Generate two-sentence weather/scene captions offline')
parser.add_argument('--weather-root', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--model', required=True, help='Local Qwen2.5-VL model directory or HF identifier')
parser.add_argument('--attention', default='flash_attention_2')
args = parser.parse_args()
weather_root, output_json = args.weather_root, args.output
selected_ids = None

# ====== 加载 Qwen 模型和处理器 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(">> Loading Qwen2.5-VL-32B-int ...")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    attn_implementation=args.attention,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(
    args.model
)
# 左侧 padding 更适合生成
processor.tokenizer.padding_side = "left"

# ====== 构建两句式 prompt ======
prompt_text = """
You are an expert aerial image analyst. Given a single aerial image, think step by step and then produce exactly two sentences in English.

Follow this reasoning process internally, but DO NOT show the steps in your answer:

1. Global weather impression: Observe the sky, lighting, color tone, and overall contrast to infer the general atmosphere (clear, overcast, foggy, rainy, snowy, nighttime, etc.).
2. Weather evidence: Look carefully for local visual cues such as raindrop streaks, snowflakes, haze, shadows, reflections, wet roads, motion blur, or low visibility that support your weather judgment.
3. Weather conclusion: Based only on visual evidence, decide one concise weather and illumination description and form the FIRST sentence. This sentence must ONLY describe the weather and illumination of the scene, without mentioning buildings, roads, or layout.

4. Scene structure under this weather: Based on the weather and visibility from the first sentence, analyze what stable man-made or natural structures can be reliably seen (buildings, fields, roads, rivers, parking lots, open spaces, vegetation, etc.), including their relative arrangement (clustered, grid-like, surrounding a field, along a road, etc.).
5. Layout reasoning: Use the weather condition to explain any visibility limitations (for example, some parts are hidden by fog or darkness) and focus on robust layout information that would remain true under different weather (relative positions, density, main axes).
6. Scene conclusion: Summarize your analysis into the SECOND sentence, which only describes the visible ground content and spatial layout (buildings, roads, open spaces, landmarks, relations). It may briefly mention visibility (for example, “though partly obscured by fog”), but MUST NOT repeat a full weather diagnosis.

Output requirements:
- Output exactly TWO plain sentences in English.
- The FIRST sentence is a pure weather and illumination description.
- The SECOND sentence describes the ground content and spatial layout, possibly mentioning visibility limitations.
- Do NOT output the reasoning steps.
- Do NOT add bullet points or numbering.
- Do NOT prefix sentences with “Sentence 1:” or “Sentence 2:”. Just write two normal sentences separated by a period.
"""

template = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": None},
            {"type": "text", "text": prompt_text},
        ],
    }
]

chat_prompt = processor.apply_chat_template(
    template,
    tokenize=False,
    add_generation_prompt=True
)

print(">> Prompt prepared.")

# ====== 主生成流程 ======

captions = {}

# 遍历每个 class 文件夹
all_class_ids = sorted(
    [d for d in os.listdir(weather_root) if os.path.isdir(os.path.join(weather_root, d))]
)

if selected_ids is not None:
    # 只保留用户指定的 ID
    all_class_ids = [cid for cid in all_class_ids if cid in selected_ids]

print(">> Found class_ids:", all_class_ids)

for class_id in tqdm(all_class_ids, desc="Classes"):
    class_dir = os.path.join(weather_root, class_id)
    if not os.path.isdir(class_dir):
        continue

    # 收集本类下所有天气图像
    img_names = sorted(
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not img_names:
        continue

    # 批量载入 PIL 图片
    images = []
    valid_names = []
    for name in img_names:
        img_path = os.path.join(class_dir, name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image: {img_path} | {e}")
            continue
        images.append(img)
        valid_names.append(name)

    if not images:
        continue

    # 构造 batch 输入
    inputs = processor(
        text=[chat_prompt] * len(images),
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 生成 caption
    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=128,
            max_length=inputs["input_ids"].size(1) + 128
        )

    # 截去 prompt 部分，只保留新生成的 token
    input_len = inputs["input_ids"].size(1)
    trimmed = outputs[:, input_len:]
    captions_list = processor.batch_decode(trimmed, skip_special_tokens=True)

    # 按 weather 存入字典
    captions[class_id] = {}
    for name, cap in zip(valid_names, captions_list):
        # 从文件名推 weather 名（例如 0839-fog.jpg -> fog）
        weather = name.rsplit("-", 1)[-1].rsplit(".", 1)[0]
        captions[class_id][weather] = cap.strip()

# 保存到 JSON
os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(captions, f, ensure_ascii=False, indent=2)

print(f"✅ Saved captions to {output_json}")
