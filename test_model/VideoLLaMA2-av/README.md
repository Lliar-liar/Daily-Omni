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
- Run `python testmodel.py` 调整`--modal-type`参数以测试不同模态下模型表现（默认为AV模态）
- 在运行之前，修改video_base_dir(视频文件目录)，json_file_path(QA_json)