"""Offline transport and scoring checks; these never call the real gateway."""

import argparse
import base64
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from pathlib import Path
import re
import tempfile
import unittest

import httpx
from openai import AsyncOpenAI

from evaluate_wita import Evaluation, PARAMETERS, retry_delay


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_scoring_and_resume_preserve_original_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            video = b"unchanged MP4 bytes including original audio"
            for index in range(4):
                video_dir = root / "Videos" / str(index)
                video_dir.mkdir(parents=True)
                (video_dir / f"{index}_video.mp4").write_bytes(video)
                items.append({
                    "Question": f"question-{index}", "Choice": ["A. one", "B. two", "C. three", "D. four"],
                    "Answer": "B", "Type": "Reasoning", "video_id": str(index),
                    "video_category": "Education", "video_duration": "30s",
                })
            args = argparse.Namespace(
                output_dir=root, video_dir=root / "Videos", max_attempts=3,
                model="AGIBOT WITA-Omni 1.0", concurrency=2, preflight=1,
            )
            counts = Counter()
            resumed = False

            async def gateway(request):
                body = json.loads(request.content)
                self.assertEqual({key: body[key] for key in PARAMETERS}, PARAMETERS)
                self.assertEqual(body["model"], args.model)
                content = body["messages"][1]["content"]
                self.assertEqual(content[0]["type"], "video_url")
                url = content[0]["video_url"]["url"]
                self.assertTrue(url.startswith("data:video/mp4;base64,"))
                self.assertEqual(base64.b64decode(url.split(",", 1)[1]), video)
                index = int(re.search(r"question-(\d+)", content[1]["text"]).group(1))
                counts[index] += 1
                if index == 0 and counts[index] == 1:
                    return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"message": "wait"}})
                if index == 3 and not resumed:
                    return httpx.Response(400, json={"error": {"message": "unsupported sample"}})
                return httpx.Response(200, json={
                    "id": f"mock-{index}", "created": 1, "model": args.model, "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "?" if index == 1 else "B"}, "finish_reason": "stop"}],
                })

            async with AsyncOpenAI(
                api_key="test-placeholder", base_url="https://example.invalid/v1", max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(gateway)),
            ) as client:
                summary = await Evaluation(args, items, client, "test-placeholder").run()
                self.assertEqual(counts, {0: 2, 1: 1, 2: 1, 3: 1})
                self.assertEqual(summary["overall"]["correct"], 2)
                self.assertEqual(summary["overall"]["valid_api_responses"], 3)
                self.assertEqual(summary["overall"]["invalid_answers"], 1)
                self.assertEqual(summary["overall"]["api_failures"], 1)
                self.assertAlmostEqual(summary["overall"]["accuracy_percent"], 200 / 3)
                self.assertEqual(summary["overall"]["accuracy_all_items_percent"], 50)
                self.assertFalse(summary["complete"])

                resumed = True
                summary = await Evaluation(args, items, client, "test-placeholder").run()
                # Neither wrong/invalid answers nor successful answers are retried.
                self.assertEqual(counts, {0: 2, 1: 1, 2: 1, 3: 2})
                self.assertEqual(summary["overall"]["accuracy_percent"], 75)
                self.assertTrue(summary["complete"])
                saved = [json.loads(line) for line in (root / "item_results.jsonl").read_text().splitlines()]
                self.assertEqual([row["item_index"] for row in saved], list(range(4)))
                self.assertEqual((root / "failures.jsonl").read_text(), "")
                self.assertNotIn("test-placeholder", (root / "journal.jsonl").read_text())

    def test_retry_after_honors_long_waits_and_http_dates(self):
        self.assertEqual(retry_delay(httpx.Headers({"Retry-After": "150"}), 1), 150)
        future = datetime.now(timezone.utc) + timedelta(seconds=120)
        delay = retry_delay(httpx.Headers({"Retry-After": format_datetime(future)}), 1)
        self.assertGreater(delay, 118)
        self.assertLessEqual(delay, 120)
        fallback = retry_delay(httpx.Headers({"Retry-After": "invalid"}), 4)
        self.assertGreaterEqual(fallback, 8)
        self.assertLessEqual(fallback, 9)


if __name__ == "__main__":
    unittest.main()
