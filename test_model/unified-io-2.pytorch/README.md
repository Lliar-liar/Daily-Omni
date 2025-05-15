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
Download the videos and revise the `video_base_dir` in `testmodel.py`
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




