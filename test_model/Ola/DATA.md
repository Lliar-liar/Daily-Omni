# Data

### Data Format

We follow the data format below, which is similar to LLaVA. You can directly use the original file path or pack the multi-modal files into patches following [create_patch.py](https://github.com/Ola-Omni/Ola/blob/main/tools/create_patch.py). Patch is a binary file containing continuous image or video files in byte format, which may accelerate reading speed in some cases.


- Image Data:

```
[
    {
        'id': ID of the data
        'image': ***.png (path to the image file or positions in patches)
        'conversations': [{"from": "human", "value": "<image>\n"}, {"from": "gpt", "value": ""}]
    }
]
```

The format for image patch is:

```
{
    "patch": "patch_00000",
    "start_num": 846113989,
    "size": 27141
}
```

- Video Frame Data:

```
[
    {
        'id': ID of the data
        'video': ***.mp4 (path to the video file or positions in patches)
        'conversations': [{"from": "human", "value": "<image>\n"}, {"from": "gpt", "value": ""}]
    }
]
```

The format for video patch is:

```
{
    "patch": "patch_000000",
    "size": [ 5605, 8902, 7917, 5562, 9249, 8785, 8379, 10389, 10505, 10337, 8481, 8164, 5562, 8844, 10565, 8035, 7768, 8969, 5643, 10478, 7632, 10980, 9986, 3602, 2848, 7591, 10766, 7813, 5605, 9840, 9664, 5605, 7726, 4828, 8006, 5562, 9711, 7903, 9542, 10626, 8827, 11268, 11115, 1832, 11354, 9222, 3965, 10426, 10427, 7311, 9726, 7655, 10025, 5350, 10098, 10470, 4877, 10273, 9730, 10150, 5604, 7203, 9881, 2246, 11114, 3790, 5567, 10490, 4072, 1701],
    "start_num": 26608266
}
```

- Video + Audio Data:

```
[
    {
        'id': ID of the data
        'video': ***.mp4 (path to the video file or positions in patches)
        'audio': ***.wav (path to the audio file)
        'conversations': [{"from": "human", "value": "<speech><image>\n"}, {"from": "gpt", "value": ""}]
    }
]
```

- Image + Audio Data:

```
[
    {
        'id': ID of the data
        'audio_q': ***.wav (path to the audio file)
        'image': ***.png (path to the image file or positions in patches)
        'conversations': [{"from": "human", "value": ""<image>\nUser's question in speech: <speech>""}, {"from": "gpt", "value": ""}]
    }
]
```

- Audio Data:

```
[
    {
        'id': ID of the data
        'audio': ***.wav (path to the audio file)
        'conversations': [{"from": "human", "value": "<speech>\n"}, {"from": "gpt", "value": ""}]
    }
]
```

### Instruction for Ola Data

**You can simply mix up the separated training jsons for joint training with image/video/audio data.**

#### **Ola-Video-1.9M**

1. Download [Ola-video-1.9M.json](https://huggingface.co/datasets/THUdyh/Ola-Data/blob/main/video_data/video-data.json) from huggingface. 

2. Download all the [video patches](https://huggingface.co/datasets/THUdyh/Ola-Data/tree/main/video_data) from huggingface. 

3. Check and modify the video patch path in the json to the true path in your machine.

#### **Ola-Audio-1.1M**

1. Download [Ola_audio_1169k.json](https://huggingface.co/datasets/THUdyh/Ola-Data/blob/main/Ola_audio_1169k.json) from huggingface. 

2. Download [wav tar file](https://huggingface.co/datasets/THUdyh/Ola-Data/tree/main/ola_audio) from huggingface and unzip all the files. 

3. Check the file structure:

```
│ola_audio/
├── Ola_audio_1169k.json
├── AudioCaps/
├── Clotho/
├── GigaSpeech/
├── LibriSpeech/
├── MillionSongDatasetSpotify/
├── MusicCaps/
├── WavCaps/
```

4. Check and modify the audio file path in the json to the true path in your machine.

#### **Ola-Cross-Modality-298k**

1. Download [Ola_cross_modality_finevideo_175k.json](https://huggingface.co/datasets/THUdyh/Ola-Data/blob/main/Ola_cross_modality_finevideo_175k.json) and [Ola_cross_modality_llava_123k.json](https://huggingface.co/datasets/THUdyh/Ola-Data/blob/main/Ola_cross_modality_llava_123k.json)  from huggingface. 

2. Download [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo/tree/main) from huggingface.

3. Download [LLaVA-Video-178k](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main) from huggingface.

4. Extract pure video from FineVideo and LLaVA-Video-178k. 

5. Transfer and save the wav file of the videos using [convert_mp4_wav.py](https://github.com/Ola-Omni/Ola/blob/main/tools/convert_mp4_wav.py).

6. Check the file structure:

```
│ola_cross_modality_298k/
├── Ola_cross_modality_finevideo_175k.json
├── Ola_cross_modality_llava_123k.json
├── finevideo_audios/
│  ├── lltmlYR56dI.wav
│  ├── ......
├── finevideo_videos/
│  ├── lltmlYR56dI.mp4
│  ├── ......
├── llava_audios/
│  ├── academic_source
│  ├── ActivityNet-QA
│  ├── liwei_youtube_videos
│  ├── NextQA
│  ├── perception_test
├── llava_videos/
│  ├── academic_source
│  ├── ActivityNet-QA
│  ├── liwei_youtube_videos
│  ├── NextQA
│  ├── perception_test
```

7. Check and modify the video and audio path in the json to the true path in your machine.

