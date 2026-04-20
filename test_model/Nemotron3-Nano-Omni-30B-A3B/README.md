# Nemotron3-Nano-Omni-30B-A3B

Evaluates `Nemotron3-Nano-Omni-30B-A3B` on the [Daily-Omni](https://huggingface.co/datasets/liarliar/Daily-Omni) QA benchmark (1197 items, audio+video+text).

## Model

- **Model**: [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning)
- **Architecture**: `NemotronH_Nano_VL_V2` (30B total, 3B active parameters)
- **Modalities**: video + audio + text
- **Inference**: `vllm serve` server + OpenAI-compatible API (`trust_remote_code=True`)
- **Thinking control**: `/no_think` system prompt + `enable_thinking=False`


## Requirements

A vLLM container with `NemotronH_Nano_VL_V2` support is available on NGC:
```
registry.ngc.nvidia.com/0767305323357365/n3-nano-omni/nemotron-3-nano-omni-reasoning-30b-a3b
```

Key dependencies (included in the container): `vllm`, `openai`, `transformers`, `soundfile`, `pandas`.

## Usage

### Usage
```bash
python testmodel.py \
  --model_name_or_path nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning \
  --video_base_dir /path/to/Videos \
  --json_file_path /path/to/qa.json \
  --input_mode all \
  --use_vllm \
  --vllm_tensor_parallel_size 8 \
  --vllm_temperature 1.0 \
  --vllm_top_k 1 \
  --max_new_tokens 1024 \
  --item_results_path runs/results_all.jsonl
```


## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_base_dir` | `/data/Videos` | Root directory of video folders (`{id}/{id}_video.mp4`) |
| `--json_file_path` | `/data/test_model/QA_all.json` | QA pairs JSON |
| `--input_mode` | `all` | `all` (video+audio) / `visual` / `audio` |
| `--model_name_or_path` | (default checkpoint above) | Model or local checkpoint path |
| `--use_vllm` | off | Use vLLM server backend (recommended) |
| `--max_new_tokens` | `1024` | Max generated tokens |
| `--vllm_tensor_parallel_size` | `0` (auto) | Number of GPUs for tensor parallelism |
| `--vllm_max_model_len` | `32768` | vLLM max sequence length |
| `--vllm_gpu_memory_utilization` | `0.95` | vLLM GPU memory fraction |
| `--vllm_temperature` | `1.0` | Sampling temperature |
| `--vllm_top_p` | `1.0` | Top-p nucleus sampling |
| `--vllm_top_k` | `1` | Top-k (`-1` = disabled) |
| `--vllm_video_fps` | `2.0` | Video frame sampling rate |
| `--vllm_long_video_fps` | `1.0` | Frame sampling override for 60s videos |
| `--vllm_long_video_max_frames` | `128` | Max frames per video |
| `--item_results_path` | auto-generated | Output JSONL path for per-item results |
| `--save_raw_output` | true | Include raw model output in JSONL |

## Output

Per-item results saved as both JSONL and XLSX under `runs/`:
- `item_results_all_<timestamp>.jsonl` / `.xlsx`

Each record includes: `video_id`, `question`, `choices`, `correct_answer`, `predicted_answer`, `is_correct`, `raw_output`, `qa_type`, `video_category`, `video_duration`, `input_mode`, `isl` (input tokens), `osl` (output tokens).

## Known Issues / History

- **vLLM offline mode**: `AssertionError: Expected code to be unreachable, got 'video'` — model plugin has no offline `LLM.generate()` support for video. Fixed by using `vllm serve` server mode.
- **`--limit-mm-per-prompt`**: `video=1,audio=1` syntax invalid for this vLLM version — removed.
- **Disk quota**: vLLM writes caches to home dir by default. Fixed by redirecting `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, `HF_HOME`, `TMPDIR` to `.cache/`.
- **Thinking not disabled**: Without `/no_think` + `enable_thinking=False`, model generates verbose chain-of-thought and accuracy collapses. Both must be set for non-reasoning mode.
