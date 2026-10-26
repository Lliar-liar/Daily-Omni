# AGIBOT WITA-Omni 1.0

Run from the repository root with Python 3.11 or newer and `openai` / `httpx`
installed. Download `Videos.tar` and `qa.json` from
[the official dataset](https://huggingface.co/datasets/liarliar/Daily-Omni), and
extract `Videos/` in the repository root. The original MP4 files include AAC
audio; the evaluator uploads the complete, unchanged files as Base64 data URLs.

```bash
python test_model_api/evaluate_wita.py \
  --output-dir runs/wita_omni_1_0 \
  --concurrency 20
```

Enter the API token at the hidden prompt, or supply it through `WITA_API_KEY`.
Do not put credentials in source files or command-line arguments. Defaults are:

- Base URL: `https://wita-omni-gateway.agibot.com/v1`
- Model: `AGIBOT WITA-Omni 1.0`
- `temperature=0`, `top_p=1`, `max_tokens=128`
- Three sequential preflight questions, then 20 concurrent requests
- Up to eight attempts for connection failures, HTTP 408/409/429 and 5xx

HTTP 429 pauses new requests according to `Retry-After` (seconds or an HTTP
date). Retries use exponential backoff when that header is absent or invalid.
The SDK's internal retries are disabled so the journal records every attempt.
The first response with choices is scored; incorrect or unparseable answers
are not retried. Answers come from `response.choices[0].message.content`.

Re-run the same command to resume. Completed API responses are preserved;
unfinished questions and API failures are retried. The evaluator checks the
dataset hash, prompt, model, endpoint and generation parameters before resuming.
Use a separate output directory for a different configuration, including a
limited run with `--max-items`.

The system prompt, user prompt and choice extraction follow the existing
`ask_gemini_av` evaluator in `test_utils.py`. Only the question, choices and
video are sent to the model. Ground-truth answers are used locally for scoring.

Outputs in the selected directory:

- `config.json`: exact prompts, parameters, input hash and software versions
- `journal.jsonl`: append-only completed responses, including previous failed attempts
- `item_results.jsonl`: one current result per question, including raw output and video hash
- `events.jsonl`: request attempts and retry timing, without credentials or video content
- `summary.json`: overall, question-type, duration and video-category scores
- `failures.jsonl`: remaining API failures

Accuracy follows the repository convention: API failures are excluded from the
response-based denominator, while unparseable API answers count as incorrect.
The summary also records coverage and accuracy over all requested questions.
A run with pending questions or API failures is not marked complete.

Run the offline request, retry, scoring and resume checks with:

```bash
python -m unittest discover -s test_model_api -p 'test_wita_evaluation.py' -v
```
