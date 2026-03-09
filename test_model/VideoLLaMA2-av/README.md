# VideoLLaMA2
## 🛠️ Requirements and Installation
Basic Dependencies:

* Pytorch >= 2.2.0
* CUDA Version >= 11.8
* transformers == 4.42.3
* tokenizers == 0.19.1


**[Offline Mode]** Install VideoLLaMA2 as a Python package (better for direct use):
```bash
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2
cd VideoLLaMA2
git checkout audio_visual
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn==2.5.8 --no-build-isolation
pip install opencv-python==4.5.5.64
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# 可能还需要安装decord之类的包
```

### Audio-Visual Checkpoints
| Model Name     | Type | Audio Encoder | Language Decoder |
|:-------------------|:----------------|:----------------|:------------------|
| [VideoLLaMA2.1-7B-AV](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-AV)  | Chat | [Fine-tuned BEATs_iter3+(AS2M)(cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) | [VideoLLaMA2.1-7B-16F](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-16F)  |



## Inference
- model checkpoints自动从huggingface下载
- Run `python testmodel.py --input_mode all`
- `--input_mode` 统一为 `{all,visual,audio}`，默认 `all`
- `all` 表示音视频联合，`visual` 表示仅视频，`audio` 表示仅音频
- `--use_audio_in_video` 已删除
- 逐条结果默认保存 raw output；如不显式传 `--item_results_path`，会写到 `runs/videollama2_av/item_results_<input_mode>_<timestamp>.jsonl`
- 在运行之前，设置 `video_base_dir`(视频文件目录) 和 `json_file_path`(QA json)

主要参数：

- `--model-path`: VideoLLaMA2 checkpoint path or HF repo.
- `--video_base_dir`: 视频根目录。
- `--json_file_path`: QA json 路径。
- `--input_mode {all,visual,audio}`: 默认 `all`。
- `--item_results_path`: 可选 JSONL 输出路径。
- `--save_raw_output`: 默认保存每条 raw output。
