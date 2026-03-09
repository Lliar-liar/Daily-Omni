# Qwen3-VL `testmodel.py`

This folder uses `testmodel.py` to evaluate Daily-Omni QA items with Qwen3-VL and vLLM.

Example:

```bash
python testmodel.py \
  --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Instruct \
  --video_base_dir Videos \
  --json_file_path qa.json \
  --input_mode all \
  --batch_size 8
```

Main parameters:

- `--video_base_dir`: video root directory.
- `--json_file_path`: QA json path.
- `--input_mode {all,visual,audio}`: default `all`.
- `--model_name_or_path`: model or local checkpoint path.
- `--processor_name_or_path`: optional processor path, defaults to model path.
- `--batch_size`: offline vLLM batch size.
- `--fps`: fps passed in message video config.
- `--do_sample_frames`: enable frame sampling controls in processor kwargs.
- `--video_metadata_mode {auto,on,off}`: video metadata handling mode.
- `--item_results_path`: optional output JSONL path.
- `--save_raw_output`: raw output is saved by default in per-item JSONL.

Useful vLLM parameters:

- `--vllm_tensor_parallel_size`, `--vllm_max_num_seqs`, `--vllm_max_model_len`, `--vllm_gpu_memory_utilization`
- `--vllm_temperature`, `--vllm_top_p`, `--vllm_top_k`, `--seed`

Notes:

- Qwen3-VL is visual-only.
- `--input_mode all` will fall back to `visual`.
- `--input_mode audio` is not supported.
