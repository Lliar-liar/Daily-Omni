

## Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignment


## Installation


#### 1. Clone this repository:
```bash
cd Ola
```

#### 2. Install the required package:
```bash
conda create -n ola python=3.10 -y
conda activate ola
pip install --upgrade pip
pip install -e .
# 使用pip==24.0
pip install decord
pip install moviepy==1.0.3
```
如果flash-attn相关错误，则`pip install flash-attn==2.5.8`（应该不会用到具体的函数，最多有import）

## Model Zoo

We provide our checkpoints at [Huggingface](https://huggingface.co/collections/THUdyh/ola-67b8220eb93406ec87aeec37)

| Model | Link | Size | Modal |
|:---:|:---:|:---:|:---:|
|Ola-7b | [Huggingface](https://huggingface.co/THUdyh/Ola-7b) | 7B | Text, Image, Video, Audio |
|Ola-Image | [Huggingface](https://huggingface.co/THUdyh/Ola-Image) | 7B | Text, Image |
|Ola-Video | [Huggingface](https://huggingface.co/THUdyh/Ola-Video) | 7B | Text, Image, Video |


## Quick Start

1. Download `Ola-7b` from [Huggingface](https://huggingface.co/THUdyh/Ola-7b) or skip the step to using the online weights directly.

2. Download audio encoder from [Huggingface](https://huggingface.co/THUdyh/Ola_speech_encoders/tree/main) and put the weights `large-v3.pt` and `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` under repo directory `/Ola/`

3. Run `inference/testmodel.py` 在运行之前，修改`video_base_dir`(视频文件目录)，`json_file_path`(QA_json)

4. `--USE_SPEECH`参数在设置为`True`时测试audio_visual，设置为`False`

## Local `inference/testmodel.py` parameters

This folder uses `inference/testmodel.py`.

Example:

```bash
python inference/testmodel.py \
  --model_path THUdyh/Ola-7b \
  --video_base_dir Videos \
  --json_file_path qa.json
```

Main parameters:

- `--model_path`: model or local checkpoint path.
- `--video_base_dir`: video root directory.
- `--json_file_path`: QA json path.
- `--text`: prompt text for direct single-sample usage.
- `--audio_path`: optional standalone audio path.
- `--image_path`: optional image path.

Note:

- Ola currently uses its own script interface and has not been aligned to the unified `--input_mode {all,visual,audio}` CLI used by the other `test_model/*/testmodel.py` scripts.
