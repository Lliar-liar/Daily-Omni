import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

from typing import List, Dict, Any, Optional, Tuple
import sys
import argparse
import json
import tqdm
import os
import re
import time


def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None


def get_video_path(video_id, base_path):
    if not base_path:
        raise ValueError("Video base path cannot be empty.")
    return os.path.join(base_path, video_id, f'{video_id}_video.mp4')


def get_audio_path(video_id, base_path):
    if not base_path:
        raise ValueError("Video base path cannot be empty.")
    return os.path.join(base_path, video_id, f'{video_id}_audio.wav')


def load_video_frames(video_path: str, fps: float = 2.0, max_frames: int = 64) -> np.ndarray:
    """Load video frames as uint8 numpy array (T, H, W, C)."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())
        step = max(1, int(round(video_fps / fps)))
        indices = list(range(0, total_frames, step))
        if len(indices) > max_frames:
            # Uniformly subsample
            idx = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = [indices[i] for i in idx]
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        return frames
    except ImportError:
        pass

    # Fallback: cv2
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / fps)))
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames) if frames else np.zeros((1, 224, 224, 3), dtype=np.uint8)


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio as float32 numpy array (samples,) and sample rate."""
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def get_effective_input_mode(args):
    return args.input_mode


def get_video_sampling_overrides(args, video_duration):
    if not getattr(args, "use_vllm", False):
        return None
    if video_duration != "60s":
        return None
    return {
        "fps": args.vllm_long_video_fps,
        "max_frames": args.vllm_long_video_max_frames,
        "min_frames": args.vllm_long_video_min_frames,
    }


def build_conversation(media_paths, question, choices, input_mode, video_overrides=None):
    if input_mode == "audio":
        media_desc = "given audio"
        user_content = [{"type": "audio", "audio": media_paths["audio_path"]}]
    elif input_mode == "all":
        media_desc = "given video and audio together"
        video_content = {"type": "video", "video": media_paths["video_path"]}
        if video_overrides:
            video_content.update(video_overrides)
        user_content = [
            video_content,
            {"type": "audio", "audio": media_paths["audio_path"]},
        ]
    else:
        media_desc = "given video"
        video_content = {"type": "video", "video": media_paths["video_path"]}
        if video_overrides:
            video_content.update(video_overrides)
        user_content = [video_content]

    choices_str = choices if isinstance(choices, str) else "\n".join(choices)
    candidate_letters = ["A", "B", "C", "D"]
    n = len(choices) if isinstance(choices, list) else 4
    letters = candidate_letters[:n]
    all_but_last = ",".join(letters[:-1])
    last = letters[-1]
    prompt = (
        f"{question}\n"
        f"{choices_str}\n"
        f"Your replies must contain only a single letter (either {all_but_last} or {last})."
    )

    system_text = "/no_think"
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": user_content + [{"type": "text", "text": prompt}],
        },
    ]


def extract_choice_letter(text):
    if not text:
        return None
    s = text.strip()
    if not s:
        return None

    # For reasoning outputs: answer after </think> tag is most reliable
    think_split = s.rsplit("</think>", 1)
    if len(think_split) == 2:
        after_think = think_split[1].strip()
        m = re.search(r"\b([ABCD])\b", after_think, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Look for explicit final-answer patterns — use \b to avoid matching
    # letters inside words (e.g. "answer because" -> 'b' in "because")
    for pattern in [
        r"(?:the\s+)?(?:correct\s+)?(?:best\s+)?answer\s+is\s*[:\s]\s*\b([ABCD])\b",
        r"(?:the\s+)?(?:correct\s+)?(?:best\s+)?answer\s*[:\s]\s*\b([ABCD])\b",
        r"(?:option|choice)\s+\b([ABCD])\b\s+is\s+(?:correct|right|best)",
        r"^\s*([ABCD])[^a-z]",  # starts with a letter
    ]:
        m = re.search(pattern, s, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Fall back to last [ABCD] occurrence (model often states answer at end)
    matches = list(re.finditer(r"\b([ABCD])\b", s))
    if matches:
        return matches[-1].group(1).upper()

    # Last resort: any [ABCD] character
    matches = list(re.finditer(r"[ABCD]", s))
    return matches[-1].group(0) if matches else None


def _extract_media_paths(conversation):
    video_path = None
    audio_path = None
    for msg in conversation:
        for part in msg.get("content", []):
            if part.get("type") == "video" and video_path is None:
                video_path = part["video"]
            elif part.get("type") == "audio" and audio_path is None:
                audio_path = part["audio"]
    return video_path, audio_path


def _encode_file_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _conversation_to_openai_messages(conversation, input_mode):
    """Convert our conversation dict to OpenAI API messages using file:// URIs."""
    messages = []
    for msg in conversation:
        role = msg["role"]
        content_parts = []
        for part in msg.get("content", []):
            if part["type"] == "text":
                content_parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "video":
                content_parts.append({
                    "type": "video_url",
                    "video_url": {"url": f"file://{part['video']}"},
                })
            elif part["type"] == "audio":
                audio_b64 = _encode_file_b64(part["audio"])
                content_parts.append({
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                })
        messages.append({"role": role, "content": content_parts})
    return messages


def generate_answer_vllm(client, conversation, args):
    """Call the vLLM OpenAI-compatible server."""
    messages = _conversation_to_openai_messages(conversation, args.input_mode)

    extra_body = {
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,
    }
    if getattr(args, "vllm_top_k", -1) > 0:
        extra_body["top_k"] = args.vllm_top_k

    try:
        response = client.chat.completions.create(
            model=args.model_name_or_path,
            messages=messages,
            max_tokens=args.max_new_tokens,
            temperature=args.vllm_temperature,
            top_p=args.vllm_top_p,
            extra_body=extra_body,
        )
        pred = response.choices[0].message.content or ""
        isl = response.usage.prompt_tokens if response.usage else None
        osl = response.usage.completion_tokens if response.usage else None
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e
    return extract_choice_letter(pred), pred, isl, osl


def generate_answer_transformers(model, processor, conversation, args):
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    video_path, audio_path = _extract_media_paths(conversation)
    video_frames = load_video_frames(video_path, fps=args.vllm_video_fps, max_frames=args.vllm_long_video_max_frames) if video_path else None
    audio_array, sample_rate = load_audio(audio_path) if audio_path else (None, None)

    inputs = processor(
        text=text,
        videos=video_frames,
        audio=audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                inputs[key] = value.to(device=model.device, dtype=model.dtype)
            else:
                inputs[key] = value.to(device=model.device)

    gen_out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    text_ids = gen_out[:, input_len:]
    decoded_text = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return extract_choice_letter(decoded_text), decoded_text, input_len, text_ids.shape[1]


def is_engine_dead_error(exc):
    name = type(exc).__name__
    if name == "EngineDeadError":
        return True
    msg = str(exc).lower()
    return "enginedeaderror" in msg or "enginecore encountered an issue" in msg


def evaluate_answer(model_answer, correct_answer):
    if not model_answer:
        return False
    return model_answer.strip().upper() == correct_answer.strip().upper()


def save_item_results_jsonl(results, output_path):
    if not output_path:
        return None
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sorted_results = sorted(
        results,
        key=lambda x: (x.get("item_index", 10**12), str(x.get("video_id", ""))),
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for record in sorted_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def save_item_results_xlsx(results, output_path):
    if not output_path:
        return None
    import pandas as pd
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sorted_results = sorted(
        results,
        key=lambda x: (x.get("item_index", 10**12), str(x.get("video_id", ""))),
    )
    df = pd.DataFrame(sorted_results)
    df.to_excel(output_path, index=False)
    return output_path


def test_all_questions(model, processor, args, sampling_params=None):
    qa_type_count = {}
    qa_type_correct = {}
    video_cat_count = {}
    video_cat_correct = {}

    data = load_json_data(args.json_file_path)
    if not data:
        print(f"Failed to load data from {args.json_file_path}. Exiting.")
        return

    total_questions = len(data)
    correct_answers = 0
    failed = 0
    VIDEO_CAT = []
    QA_TYPE = []
    item_results = []

    for item in data:
        video_category = item.get('video_category') or "Unknown"
        qa_type = item.get('Type') or "Unknown"
        if video_category not in VIDEO_CAT:
            VIDEO_CAT.append(video_category)
        if qa_type not in QA_TYPE:
            QA_TYPE.append(qa_type)

    VIDEO_CAT.sort()
    QA_TYPE.sort()

    for qa_type in QA_TYPE:
        qa_type_count[qa_type] = 0
        qa_type_correct[qa_type] = 0
    for video_category in VIDEO_CAT:
        video_cat_count[video_category] = 0
        video_cat_correct[video_category] = 0

    total_questions = len(data)
    correct_answers = 0
    failed = 0
    qa_duration_count = {"30s": 0, "60s": 0}
    qa_duration_correct = {"30s": 0, "60s": 0}

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using video base directory: {args.video_base_dir}")
    print(f"Input mode: {args.input_mode}")
    effective_input_mode = get_effective_input_mode(args)

    def append_item_result(
        item_meta,
        *,
        predicted_answer=None,
        raw_output="",
        is_correct=False,
        api_call_failed=False,
        skipped=False,
        reason=None,
        isl=None,
        osl=None,
    ):
        record = {
            "item_index": item_meta.get("idx"),
            "video_id": item_meta.get("video_id"),
            "question": item_meta.get("question"),
            "choices": item_meta.get("choices"),
            "correct_answer": item_meta.get("correct_answer"),
            "predicted_answer": predicted_answer,
            "is_correct": bool(is_correct),
            "api_call_failed": bool(api_call_failed),
            "skipped": bool(skipped),
            "reason": reason,
            "qa_type": item_meta.get("qa_type"),
            "video_category": item_meta.get("video_category"),
            "video_duration": item_meta.get("video_duration"),
            "input_mode": item_meta.get("input_mode"),
            "isl": isl,
            "osl": osl,
        }
        if args.save_raw_output:
            record["raw_output"] = raw_output
        item_results.append(record)

    for idx, item in enumerate(tqdm.tqdm(data, desc="Evaluating Questions")):
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')
        qa_type = item.get('Type')
        video_category = item.get('video_category')
        video_duration = item.get('video_duration')
        base_item_meta = {
            "idx": idx,
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "qa_type": qa_type,
            "video_category": video_category,
            "video_duration": video_duration,
            "input_mode": effective_input_mode,
        }

        # Only require fields needed for inference; category/duration are reporting-only
        if not all([question, choices, correct_answer, video_id]):
            print(f"\nWarning: Skipping item due to missing fields. Item Index: {idx}, Video ID: {video_id or 'Unknown'}")
            failed += 1
            append_item_result(base_item_meta, skipped=True, reason="missing_fields")
            continue
        qa_type = qa_type or "Unknown"
        video_category = video_category or "Unknown"
        video_duration = video_duration or "30s"

        try:
            if effective_input_mode == "audio":
                audio_path = get_audio_path(video_id, args.video_base_dir)
                if not os.path.exists(audio_path):
                    print(f"\nWarning: Audio file not found for ID {video_id} at {audio_path}. Skipping.")
                    failed += 1
                    append_item_result(base_item_meta, skipped=True, api_call_failed=True, reason=f"audio_not_found:{audio_path}")
                    continue
                media_paths = {"audio_path": audio_path}
            elif effective_input_mode == "all":
                video_path = get_video_path(video_id, args.video_base_dir)
                audio_path = get_audio_path(video_id, args.video_base_dir)
                missing = []
                if not os.path.exists(video_path):
                    missing.append(f"video={video_path}")
                if not os.path.exists(audio_path):
                    missing.append(f"audio={audio_path}")
                if missing:
                    print(f"\nWarning: Missing media for ID {video_id}: {', '.join(missing)}. Skipping.")
                    failed += 1
                    append_item_result(base_item_meta, skipped=True, api_call_failed=True, reason=f"missing_media:{','.join(missing)}")
                    continue
                media_paths = {"video_path": video_path, "audio_path": audio_path}
            else:
                video_path = get_video_path(video_id, args.video_base_dir)
                if not os.path.exists(video_path):
                    print(f"\nWarning: Video file not found for ID {video_id} at {video_path}. Skipping.")
                    failed += 1
                    append_item_result(base_item_meta, skipped=True, api_call_failed=True, reason=f"video_not_found:{video_path}")
                    continue
                media_paths = {"video_path": video_path}
        except ValueError as e:
            print(f"\nError constructing media path: {e}. Skipping item for video ID {video_id}")
            failed += 1
            append_item_result(base_item_meta, skipped=True, api_call_failed=True, reason=f"media_path_error:{e}")
            continue

        conversation = build_conversation(
            media_paths=media_paths,
            question=question,
            choices=choices,
            input_mode=effective_input_mode,
            video_overrides=get_video_sampling_overrides(args, video_duration),
        )
        model_answer = None
        decoded_text = ""
        isl = None
        osl = None
        try:
            if args.use_vllm:
                model_answer, decoded_text, isl, osl = generate_answer_vllm(
                    client=model,
                    conversation=conversation,
                    args=args,
                )
            else:
                model_answer, decoded_text, isl, osl = generate_answer_transformers(
                    model=model,
                    processor=processor,
                    conversation=conversation,
                    args=args,
                )

            if model_answer is None:
                print(
                    f"\nWarning: Could not extract answer for video {video_id}. "
                    f"Raw output: '{decoded_text}'"
                )

        except Exception as e:
            import traceback
            print(f"\nError processing video {video_id} (Index: {idx}): {type(e).__name__}: {e!r}")
            traceback.print_exc(limit=2)
            failed += 1
            append_item_result(
                base_item_meta,
                raw_output=decoded_text,
                api_call_failed=True,
                reason=f"inference_error:{type(e).__name__}:{e}",
                isl=isl,
                osl=osl,
            )
            continue

        normalized_model_answer = model_answer or extract_choice_letter(decoded_text)
        is_correct = evaluate_answer(normalized_model_answer, correct_answer)

        if qa_type in qa_type_count:
            qa_type_count[qa_type] += 1
            if is_correct:
                qa_type_correct[qa_type] += 1
        if video_category in video_cat_count:
            video_cat_count[video_category] += 1
            if is_correct:
                video_cat_correct[video_category] += 1
        if video_duration in qa_duration_count:
            qa_duration_count[video_duration] += 1
            if is_correct:
                qa_duration_correct[video_duration] += 1

        if is_correct:
            correct_answers += 1
        append_item_result(
            base_item_meta,
            predicted_answer=normalized_model_answer,
            raw_output=decoded_text,
            is_correct=is_correct,
            isl=isl,
            osl=osl,
        )

    print("\n--- Evaluation Summary ---")
    valid_questions = total_questions - failed
    if valid_questions > 0:
        print(f"Overall Accuracy: {correct_answers}/{valid_questions} = {correct_answers / valid_questions:.2%}")
    else:
        print("Overall Accuracy: 0/0 = N/A (No questions processed successfully)")
    print(f"(Total items: {total_questions}, Skipped/Failed items: {failed})")

    print("\n--- Accuracy by QA Type ---")
    for qa_type in QA_TYPE:
        count = qa_type_count.get(qa_type, 0)
        correct = qa_type_correct.get(qa_type, 0)
        print(f"{qa_type}: {correct}/{count} = {correct / count:.2%}" if count else f"{qa_type}: 0/0 = N/A")

    print("\n--- Accuracy by Video Category ---")
    for video_category in VIDEO_CAT:
        count = video_cat_count.get(video_category, 0)
        correct = video_cat_correct.get(video_category, 0)
        print(f"{video_category}: {correct}/{count} = {correct / count:.2%}" if count else f"{video_category}: 0/0 = N/A")

    print("\n--- Accuracy by Video Duration ---")
    for duration in ["30s", "60s"]:
        count = qa_duration_count.get(duration, 0)
        correct = qa_duration_correct.get(duration, 0)
        print(f"{duration} Duration: {correct}/{count} = {correct / count:.2%}" if count else f"{duration} Duration: 0/0 = N/A")

    print(f"\nTotal items failed during processing: {failed}")
    item_results_path = args.item_results_path
    if not item_results_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        item_results_path = os.path.join(
            "runs", "nemotron3_nano_omni", f"item_results_{effective_input_mode}_{ts}.jsonl"
        )
    written_path = save_item_results_jsonl(item_results, item_results_path)
    if written_path:
        print(f"Per-item results written to: {written_path}")
    xlsx_path = item_results_path.replace(".jsonl", ".xlsx") if item_results_path else None
    written_xlsx = save_item_results_xlsx(item_results, xlsx_path)
    if written_xlsx:
        print(f"Per-item results (xlsx) written to: {written_xlsx}")
    print("--- Evaluation Complete ---")


def load_vllm_backend(args):
    """Start a vLLM OpenAI-compatible server and return an openai.Client pointed at it."""
    import subprocess
    import socket
    from openai import OpenAI

    tp_size = args.vllm_tensor_parallel_size
    if tp_size <= 0:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible_devices:
            tp_size = max(1, len([x for x in cuda_visible_devices.split(",") if x.strip()]))
        else:
            tp_size = 1

    # Pick a free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    import json as _json
    media_io_kwargs = _json.dumps({"video": {"num_frames": args.vllm_num_video_frames, "fps": args.vllm_video_fps}})
    cmd = [
        "vllm", "serve", args.model_name_or_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--max-model-len", str(args.vllm_max_model_len),
        "--max-num-seqs", str(args.vllm_max_num_seqs),
        "--dtype", "bfloat16",
        "--allowed-local-media-path", "/",
        "--mamba-ssm-cache-dtype", "float32",
        "--media-io-kwargs", media_io_kwargs,
    ]

    print(f"Starting vLLM server on port {port} with TP={tp_size}...")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    server_proc = subprocess.Popen(cmd, env=env)

    # Wait for server to be ready
    import urllib.request
    health_url = f"http://localhost:{port}/health"
    print(f"Waiting for vLLM server to be ready at {health_url} ...")
    for _ in range(300):  # up to 5 minutes
        time.sleep(2)
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print("vLLM server is ready.")
            break
        except Exception:
            if server_proc.poll() is not None:
                raise RuntimeError(f"vLLM server process exited with code {server_proc.returncode}")
    else:
        server_proc.terminate()
        raise RuntimeError("vLLM server did not start within 10 minutes.")

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    return client, server_proc


_DEFAULT_MODEL = (
    "/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/workspace/output/danial_rl/"
    "nano-v3-vl-comboGRPO_v3_SFT-MPO-TRL1-IRLs125-ckpt-step-25-stage2-rl/iter_125/mcore_to_hf"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Nemotron3-Nano-Omni-30B-A3B on the Daily-Omni dataset.")

    # Data
    parser.add_argument('--video_base_dir', type=str, default='/data/Videos',
                        help='Base directory containing video folders.')
    parser.add_argument('--json_file_path', type=str, default='/data/test_model/QA_all.json',
                        help='Path to the JSON file containing QA pairs.')

    # Modality
    parser.add_argument('--input_mode', type=str, default='all',
                        choices=['all', 'visual', 'audio'],
                        help='Input modality for evaluation.')

    # Model
    parser.add_argument('--model_name_or_path', type=str, default=_DEFAULT_MODEL,
                        help='HuggingFace model name or local checkpoint path.')
    parser.add_argument('--processor_name_or_path', type=str, default=None,
                        help='Processor path. Defaults to model_name_or_path if not set.')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device map for loading the model (e.g., "auto", "cuda:0").')
    parser.add_argument('--precision', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Precision for model loading.')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2',
                        choices=['flash_attention_2', 'sdpa', 'eager', 'None'],
                        help='Attention implementation.')
    parser.add_argument('--max_new_tokens', type=int, default=16384,
                        help='Maximum new tokens generated per question.')

    # vLLM backend
    parser.add_argument('--use_vllm', action='store_true',
                        help='Use vLLM backend for inference.')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=0,
                        help='0 = auto-detect from CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--vllm_max_num_seqs', type=int, default=1)
    parser.add_argument('--vllm_max_model_len', type=int, default=32768)
    parser.add_argument('--vllm_temperature', type=float, default=1.0)
    parser.add_argument('--vllm_top_p', type=float, default=1.0)
    parser.add_argument('--vllm_top_k', type=int, default=1)
    parser.add_argument('--vllm_engine_restart_retries', type=int, default=1)
    parser.add_argument('--vllm_num_video_frames', type=int, default=128,
                        help='Number of frames vLLM extracts from each video.')
    parser.add_argument('--vllm_video_fps', type=float, default=2.0,
                        help='Default video sampling fps for vLLM.')
    parser.add_argument('--vllm_long_video_fps', type=float, default=1.0,
                        help='Sampling fps override for 60s videos.')
    parser.add_argument('--vllm_long_video_min_frames', type=int, default=4)
    parser.add_argument('--vllm_long_video_max_frames', type=int, default=192)
    parser.add_argument('--seed', type=int, default=1234)

    # Output
    parser.add_argument('--item_results_path', type=str, default=None,
                        help='Path to save per-item JSONL results.')
    parser.add_argument('--save_raw_output', action='store_true', default=True,
                        help='Save raw model output text in per-item JSONL.')

    args = parser.parse_args()

    if args.vllm_engine_restart_retries < 0:
        print("Error: --vllm_engine_restart_retries must be >= 0.")
        sys.exit(1)
    if args.vllm_long_video_fps <= 0:
        print("Error: --vllm_long_video_fps must be > 0.")
        sys.exit(1)
    if args.vllm_long_video_min_frames < 1:
        print("Error: --vllm_long_video_min_frames must be >= 1.")
        sys.exit(1)
    if args.vllm_long_video_max_frames < args.vllm_long_video_min_frames:
        print("Error: --vllm_long_video_max_frames must be >= --vllm_long_video_min_frames.")
        sys.exit(1)

    if args.processor_name_or_path is None:
        args.processor_name_or_path = args.model_name_or_path

    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    torch_dtype = dtype_map.get(args.precision, torch.bfloat16)
    attn_impl = args.attn_implementation if args.attn_implementation != "None" else None

    print(f"Loading processor: {args.processor_name_or_path}...")
    try:
        processor = AutoProcessor.from_pretrained(
            args.processor_name_or_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading processor: {e}")
        sys.exit(1)

    model = None
    server_proc = None
    try:
        if args.use_vllm:
            model, server_proc = load_vllm_backend(args)
        else:
            print(f"Loading model with Transformers: {args.model_name_or_path}")
            print(f"Precision: {args.precision} | Attention: {attn_impl} | Device: {args.device}")
            load_kwargs = dict(
                device_map=args.device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            if attn_impl is not None:
                load_kwargs["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
            print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading backend: {e}")
        sys.exit(1)

    try:
        test_all_questions(model, processor, args)
    finally:
        if server_proc is not None:
            print("Shutting down vLLM server...")
            server_proc.terminate()
            server_proc.wait(timeout=30)
