# Qwen2.5-Omni `testmodel.py`

This folder uses `testmodel.py` to evaluate Daily-Omni QA items with Qwen2.5-Omni.

Example:

```bash
python testmodel.py \
  --model_name_or_path Qwen/Qwen2.5-Omni-7B-Instruct \
  --video_base_dir Videos \
  --json_file_path qa.json \
  --input_mode all \
  --use_vllm
```

Main parameters:

- `--video_base_dir`: video root directory.
- `--json_file_path`: QA json path.
- `--input_mode {all,visual,audio}`: modality selection, default `all`.
- `--model_name_or_path`: model or local checkpoint path.
- `--processor_name_or_path`: optional processor path, defaults to model path.
- `--use_vllm`: use vLLM backend.
- `--max_new_tokens`: max generated tokens per sample.
- `--item_results_path`: optional output JSONL path.
- `--save_raw_output`: raw output is saved by default in per-item JSONL.

Useful backend parameters:

- `--device`, `--precision`, `--attn_implementation`: transformers backend loading.
- `--vllm_tensor_parallel_size`, `--vllm_max_num_seqs`, `--vllm_max_model_len`, `--vllm_gpu_memory_utilization`: vLLM runtime settings.
