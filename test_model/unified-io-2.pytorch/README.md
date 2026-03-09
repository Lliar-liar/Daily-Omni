# Unified IO 2
## Installation
**Install [pytorch](https://pytorch.org/) following the recommendation for your system** .
CUDA>=11.8 should be ok.
Then install with

```
git clone unified-io-2.pytorch
cd unified-io-2.pytorch
pip install -r requirements.txt
```
Download the videos, then pass `--video_base_dir` and `--json_file_path` when running `testmodel.py`.
## Loading the model

Load the model with 
```
from uio2.model import UnifiedIOModel
model = UnifiedIOModel.from_pretrained("allenai/uio2-large")
```
This loads the large (1B) model, load the XL (3B) or XXL (7B) with 
`allenai/uio2-xl` and `allenai/uio2-xxl`.

This model requires pre-processed tensor inputs. Pre-processing is done by `UnifiedIOPreprocessor`:

```
from uio2.preprocessing import UnifiedIOPreprocessor
preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer="/path/to/tokenizer")
```

Here "/path/to/tokenizer.model" needs to point to the LLaMa tokenizer file.

## Test the model
Run `testmodel.py`. Adjust the `--model` parameter to test different model.

Example:

```bash
python testmodel.py \
  --model allenai/uio2-large \
  --video_base_dir Videos \
  --json_file_path qa.json \
  --input_mode all
```

Unified CLI notes:

- `--input_mode {all,visual,audio}` is exposed for consistency with other `test_model` scripts.
- `unified-io-2.pytorch` currently only supports `--input_mode all`.
- Raw model output is saved by default in per-item JSONL. If `--item_results_path` is omitted, output is written to `runs/unified_io_2/item_results_all_<timestamp>.jsonl`.

Main parameters:

- `--model`: model name, for example `allenai/uio2-large`.
- `--gpu`: GPU id, use `-1` for CPU.
- `--video_base_dir`: video root directory.
- `--json_file_path`: QA json path.
- `--input_mode`: keep this as `all`.
- `--item_results_path`: optional JSONL output path.
- `--save_raw_output`: raw output is saved by default.


