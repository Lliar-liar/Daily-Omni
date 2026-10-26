"""Evaluate WITA-Omni using the original MP4 bytes, including their audio track.

Uses the prompts and choice extraction from test_utils.ask_gemini_av. Credentials
come from WITA_API_KEY or a hidden terminal prompt, and are never saved.
"""

import argparse
import asyncio
import base64
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time

import httpx
import openai
from openai import AsyncOpenAI


SYSTEM_PROMPT = """
    Your task is to accurately answer multiple-choice questions based on the given video.
    Select the single most accurate answer from the given choices.
    Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
    """
USER_PROMPT = "Given the video, answer the question below. \n    Question: {question}\n    Choices: {choices}"
PARAMETERS = {"temperature": 0, "top_p": 1, "max_tokens": 128}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def extract_choice_letter(text):
    """Match the repository's existing scoring rule exactly."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    if text[0] in "ABCD":
        return text[0]
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None


def retry_delay(headers, attempt):
    """Honor Retry-After seconds or HTTP dates, without truncating server waits."""
    value = headers.get("retry-after")
    if value is not None:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(60.0, 2.0 ** (attempt - 1)) + random.uniform(0, 1)


def encode_video(path):
    data = path.read_bytes()
    return (
        "data:video/mp4;base64," + base64.b64encode(data).decode("ascii"),
        hashlib.sha256(data).hexdigest(),
        len(data),
    )


def build_messages(item, video_url):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": USER_PROMPT.format(
                    question=item["Question"], choices=item["Choice"]
                )},
            ],
        },
    ]


def summarize(items, results):
    def metrics(indices):
        rows = [results[i] for i in indices if i in results]
        answered = [r for r in rows if not r["api_call_failed"]]
        correct = sum(r["is_correct"] for r in answered)
        return {
            "requested": len(indices),
            "completed": len(rows),
            "pending": len(indices) - len(rows),
            "api_failures": sum(r["api_call_failed"] for r in rows),
            "valid_api_responses": len(answered),
            "invalid_answers": sum(r["predicted_answer"] is None for r in answered),
            "correct": correct,
            "accuracy_percent": 100 * correct / len(answered) if answered else None,
            "accuracy_all_items_percent": 100 * correct / len(indices) if indices else None,
        }

    summary = {"updated_at": now(), "overall": metrics(list(range(len(items))))}
    for label, field in [
        ("by_type", "Type"),
        ("by_duration", "video_duration"),
        ("by_video_category", "video_category"),
    ]:
        summary[label] = {
            value: metrics([i for i, q in enumerate(items) if q[field] == value])
            for value in sorted({q[field] for q in items})
        }
    summary["all_items_attempted"] = summary["overall"]["pending"] == 0
    summary["complete"] = summary["all_items_attempted"] and summary["overall"]["api_failures"] == 0
    summary["response_models"] = dict(Counter(
        r.get("response_model") for r in results.values() if not r["api_call_failed"]
    ))
    summary["usage"] = {
        field: sum((r.get("usage") or {}).get(field, 0) or 0 for r in results.values())
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    return summary


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class Evaluation:
    def __init__(self, args, items, client, api_key):
        self.args, self.items, self.client, self.api_key = args, items, client, api_key
        self.output = args.output_dir
        self.results = {}
        self.blocked_until = 0.0
        self.started = time.monotonic()
        journal = self.output / "journal.jsonl"
        if journal.exists():
            with journal.open() as source:
                for line in source:
                    row = json.loads(line)
                    self.results[row["item_index"]] = row
        self.journal = journal.open("a", encoding="utf-8", buffering=1)
        self.events = (self.output / "events.jsonl").open("a", encoding="utf-8", buffering=1)

    def event(self, **fields):
        self.events.write(json.dumps({"time": now(), **fields}, ensure_ascii=False) + "\n")

    def error_info(self, error):
        # Never stringify a request, headers, or an exception carrying the video.
        info = {"type": type(error).__name__}
        if isinstance(error, openai.APIStatusError):
            info["status_code"] = error.status_code
            try:
                body = error.response.json()
                detail = body.get("error", body) if isinstance(body, dict) else {}
                if isinstance(detail, dict):
                    for field in ["code", "type", "message"]:
                        value = str(detail.get(field, "")).replace(self.api_key, "[REDACTED]")
                        value = re.sub(r"data:video/mp4;base64,[A-Za-z0-9+/=]+", "[VIDEO]", value)
                        info["server_" + field] = value[:500]
            except (ValueError, TypeError):
                pass
        return info

    async def wait_for_rate_limit(self):
        while self.blocked_until > time.monotonic():
            await asyncio.sleep(self.blocked_until - time.monotonic())

    async def evaluate_item(self, index):
        item = self.items[index]
        path = self.args.video_dir / item["video_id"] / f'{item["video_id"]}_video.mp4'
        url, video_hash, video_bytes = await asyncio.to_thread(encode_video, path)
        messages = build_messages(item, url)
        row = {
            "item_index": index, "item_id": item["video_id"], "video_id": item["video_id"],
            "question": item["Question"], "choices": item["Choice"],
            "correct_answer": item["Answer"], "qa_type": item["Type"],
            "video_category": item["video_category"], "video_duration": item["video_duration"],
            "video_sha256": video_hash, "video_bytes": video_bytes,
            "predicted_answer": None, "api_answer": None, "raw_output": None,
            "is_correct": False, "api_call_failed": True, "skipped": False,
            "status": "api_error", "errors": [], "started_at": now(),
        }
        started = time.monotonic()
        for attempt in range(1, self.args.max_attempts + 1):
            await self.wait_for_rate_limit()
            row["attempts"] = attempt
            self.event(event="request", item_index=index, attempt=attempt, video_bytes=video_bytes)
            try:
                response = await self.client.chat.completions.create(
                    model=self.args.model, messages=messages, **PARAMETERS
                )
                if not response.choices:
                    row["errors"].append({"type": "MissingChoices", "attempt": attempt})
                    break
                choice = response.choices[0]
                answer = choice.message.content
                prediction = extract_choice_letter(answer)
                row.update({
                    "raw_output": answer, "api_answer": answer, "predicted_answer": prediction,
                    "is_correct": prediction == item["Answer"].strip().upper(),
                    "api_call_failed": False, "status": "ok" if prediction else "invalid_answer",
                    "response_id": response.id, "response_model": response.model,
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                })
                break
            except (openai.APIStatusError, openai.APIConnectionError) as error:
                info = {**self.error_info(error), "attempt": attempt}
                row["errors"].append(info)
                status = info.get("status_code")
                retryable = status in (408, 409, 429) or (status is not None and status >= 500)
                retryable = retryable or isinstance(error, openai.APIConnectionError)
                if not retryable or attempt == self.args.max_attempts:
                    self.event(event="failure", item_index=index, **info)
                    break
                headers = error.response.headers if isinstance(error, openai.APIStatusError) else {}
                delay = retry_delay(headers, attempt)
                if status == 429:
                    self.blocked_until = max(self.blocked_until, time.monotonic() + delay)
                self.event(event="retry", item_index=index, delay_seconds=delay, **info)
                print(f'Retry item={index} status={status or info["type"]} wait={delay:.1f}s', flush=True)
                await asyncio.sleep(delay)
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        row["completed_at"] = now()
        return row

    def record(self, row):
        self.results[row["item_index"]] = row
        self.journal.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        completed = len(self.results)
        if completed <= self.args.preflight or completed % 20 == 0 or completed == len(self.items):
            self.checkpoint()
            correct = sum(r["is_correct"] for r in self.results.values())
            failures = sum(r["api_call_failed"] for r in self.results.values())
            print(f'Progress {completed}/{len(self.items)}; correct={correct}; '
                  f'API failures={failures}; elapsed={time.monotonic() - self.started:.0f}s', flush=True)

    def checkpoint(self):
        summary = summarize(self.items, self.results)
        summary.update({"model": self.args.model, "parameters": PARAMETERS})
        atomic_text(self.output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        rows = [self.results[i] for i in sorted(self.results)]
        atomic_text(self.output / "item_results.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        atomic_text(self.output / "failures.jsonl", "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["api_call_failed"]
        ))
        return summary

    async def run(self):
        pending = [i for i in range(len(self.items))
                   if i not in self.results or self.results[i]["api_call_failed"]]
        print(f'{len(self.items)} questions; {len(pending)} pending; concurrency={self.args.concurrency}', flush=True)
        self.event(event="session_start", pending=len(pending), concurrency=self.args.concurrency)
        try:
            # Preflight checks transport, never selects or retries by correctness.
            for index in pending[:self.args.preflight]:
                row = await self.evaluate_item(index)
                self.record(row)
                if row["api_call_failed"]:
                    raise RuntimeError(f'Preflight failed for item {index}; see failures.jsonl')
            queue = asyncio.Queue()
            for index in pending[self.args.preflight:]:
                queue.put_nowait(index)

            async def worker():
                while not queue.empty():
                    index = queue.get_nowait()
                    row = await self.evaluate_item(index)
                    self.record(row)
                    if any(e.get("status_code") == 401 for e in row["errors"]):
                        raise RuntimeError("Authentication failed; see failures.jsonl")

            async with asyncio.TaskGroup() as group:
                for _ in range(min(self.args.concurrency, queue.qsize())):
                    group.create_task(worker())
        finally:
            self.checkpoint()
            self.journal.close()
            self.events.close()
        return summarize(self.items, self.results)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-file", type=Path, default=Path("qa.json"))
    parser.add_argument("--video-dir", type=Path, default=Path("Videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / ("wita_omni_1_0_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--base-url", default="https://wita-omni-gateway.agibot.com/v1")
    parser.add_argument("--model", default="AGIBOT WITA-Omni 1.0")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--preflight", type=int, default=3)
    parser.add_argument("--api-key-stdin", action="store_true", help="Read the token from stdin instead of the hidden prompt.")
    args = parser.parse_args()
    if min(args.concurrency, args.max_attempts, args.timeout) <= 0 or args.preflight < 0:
        parser.error("Concurrency, attempts and timeout must be positive; preflight must be nonnegative.")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    items = json.loads(args.qa_file.read_text(encoding="utf-8"))
    if args.max_items:
        items = items[:args.max_items]
    if not items:
        parser.error("No questions selected")
    for index, item in enumerate(items):
        required = ["Question", "Choice", "Answer", "Type", "video_id", "video_category", "video_duration"]
        if not all(item.get(field) for field in required) or item["Answer"] not in "ABCD":
            raise ValueError(f"Invalid question {index}")
        path = args.video_dir / item["video_id"] / f'{item["video_id"]}_video.mp4'
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty video for question {index}: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {
        "model": args.model, "base_url": args.base_url, "parameters": PARAMETERS,
        "system_prompt": SYSTEM_PROMPT, "user_prompt_template": USER_PROMPT,
        "qa_file": str(args.qa_file.resolve()), "qa_sha256": digest_file(args.qa_file),
        "video_dir": str(args.video_dir.resolve()), "items_requested": len(items),
        "input_mode": "original_mp4_with_audio_as_base64_data_url",
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        old_config = json.loads(config_path.read_text())
        if any(old_config.get(key) != value for key, value in config.items()):
            raise ValueError("Resume configuration differs; use a new output directory")
    else:
        config.update({
            "created_at": now(), "concurrency": args.concurrency, "max_attempts": args.max_attempts,
            "timeout_seconds": args.timeout, "preflight": args.preflight,
            "openai_version": openai.__version__, "httpx_version": httpx.__version__,
            "python_version": sys.version, "script_sha256": digest_file(__file__),
        })
        atomic_text(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    api_key = os.environ.get("WITA_API_KEY")
    if not api_key:
        api_key = sys.stdin.readline().strip() if args.api_key_stdin else getpass.getpass("WITA API token (hidden): ").strip()
    if not api_key:
        raise ValueError("WITA_API_KEY is empty")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    timeout = httpx.Timeout(args.timeout, connect=30, write=args.timeout, pool=60)
    async with AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=timeout, max_retries=0) as client:
        summary = await Evaluation(args, items, client, api_key).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    lock.close()
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Interrupted; completed responses are saved. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(130)
