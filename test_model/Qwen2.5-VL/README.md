# Qwen2.5-VL `testmodel.py`

This folder uses `testmodel.py` to evaluate Daily-Omni QA items with Qwen2.5-VL.

Example:

```bash
python testmodel.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --video_base_dir Videos \
  --json_file_path qa.json \
  --input_mode all \
  --fps 2
```

Main parameters:

- `--video_base_dir`: video root directory.
- `--json_file_path`: QA json path.
- `--input_mode {all,visual,audio}`: default `all`.
- `--model_name_or_path`: model or local checkpoint path.
- `--processor_name_or_path`: optional processor path, defaults to model path.
- `--fps`: sampled video fps for processor input.
- `--device`, `--precision`, `--attn_implementation`: model loading options.
- `--item_results_path`: optional output JSONL path.
- `--save_raw_output`: raw output is saved by default in per-item JSONL.

Notes:

- Qwen2.5-VL is visual-only.
- `--input_mode all` will fall back to `visual`.
- `--input_mode audio` is not supported.
