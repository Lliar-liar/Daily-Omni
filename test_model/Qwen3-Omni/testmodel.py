import argparse
import json
import os
import re
import sys
import time

import torch
import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    process_mm_info = None


def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None


def get_video_path(video_id, base_path):
    if not base_path:
        raise ValueError("Video base path cannot be empty.")
    return os.path.join(base_path, video_id, f"{video_id}_video.mp4")


def get_audio_path(video_id, base_path):
    if not base_path:
        raise ValueError("Video base path cannot be empty.")
    return os.path.join(base_path, video_id, f"{video_id}_audio.wav")


def get_effective_use_audio_in_video(_args):
    # The CLI now uses explicit modality selection only:
    # visual = video frames only, audio = audio only, all = both.
    return False


def extract_choice_letter(text):
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None

    # For Thinking models, prefer the segment after the final </think>.
    post_think = raw
    think_end_matches = list(re.finditer(r"</think>", raw, flags=re.IGNORECASE))
    if think_end_matches:
        post_think = raw[think_end_matches[-1].end() :].strip()

    # Also create a version with complete <think>...</think> blocks removed.
    no_think_blocks = re.sub(r"(?is)<think>.*?</think>", " ", raw).strip()

    candidates = []
    for candidate in [post_think, no_think_blocks, raw]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        # Explicit answer tags/phrases first.
        answer_tag = re.search(r"(?is)<answer>\s*([A-D])\s*</answer>", candidate)
        if answer_tag:
            return answer_tag.group(1).upper()

        answer_phrase = re.search(
            r"(?i)(?:^|[\n\r])\s*(?:final answer|answer|the answer is|答案)\s*[:：]?\s*([A-D])\b",
            candidate,
        )
        if answer_phrase:
            return answer_phrase.group(1).upper()

        direct_match = re.match(r"(?i)^\s*([A-D])(?:[\s\.\):：]|$)", candidate)
        if direct_match:
            return direct_match.group(1).upper()

        # Fallback: use the last standalone option letter in this candidate.
        all_matches = re.findall(r"\b([A-D])\b", candidate.upper())
        if all_matches:
            return all_matches[-1]

    return None


def evaluate_answer(model_answer, correct_answer):
    extracted = extract_choice_letter(model_answer)
    if extracted is None:
        return False
    return extracted == correct_answer.strip().upper()


def build_conversation(media_paths, question, choices, input_mode):
    content = []
    if input_mode == "audio":
        task_prompt = "given audio"
        content.append({"type": "audio", "audio": media_paths["audio_path"]})
    elif input_mode == "all":
        task_prompt = "given video and audio together"
        content.append({"type": "video", "video": media_paths["video_path"]})
        content.append({"type": "audio", "audio": media_paths["audio_path"]})
    else:
        task_prompt = "given video"
        content.append({"type": "video", "video": media_paths["video_path"]})

    prompt = (
        "Your task is to accurately answer multiple-choice questions "
        f"based on the {task_prompt}.\n"
        "Select the single most accurate answer from the given choices.\n"
        f"Question: {question}\n"
        f"Choices: {choices}\n"
        "Your answer should be a capital letter representing your choice: "
        "A, B, C, or D. Don't generate any other text.\n"
    )

    return [
        {
            "role": "user",
            "content": content + [{"type": "text", "text": prompt}],
        }
    ]


def build_vllm_inputs(processor, conversation, use_audio_in_video):
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )

    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }
    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios
    return inputs


def generate_answer_transformers(model, processor, conversation, args):
    effective_use_audio_in_video = get_effective_use_audio_in_video(args)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=effective_use_audio_in_video
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=effective_use_audio_in_video,
    )
    # Keep integer tensors (e.g., input_ids) untouched, while aligning all
    # floating tensors (audio/video features) to model dtype to avoid dtype
    # mismatch errors like float32 vs bfloat16 during generation.
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                inputs[key] = value.to(device=model.device, dtype=model.dtype)
            else:
                inputs[key] = value.to(device=model.device)

    generation_kwargs = {
        "return_audio": False,
        "use_audio_in_video": effective_use_audio_in_video,
        "thinker_return_dict_in_generate": True,
        "thinker_max_new_tokens": args.max_new_tokens,
        "thinker_do_sample": False,
    }
    outputs = model.generate(**inputs, **generation_kwargs)

    if isinstance(outputs, tuple):
        text_outputs = outputs[0]
    else:
        text_outputs = outputs

    if hasattr(text_outputs, "sequences"):
        generated_ids = text_outputs.sequences
    else:
        generated_ids = text_outputs

    input_len = inputs["input_ids"].shape[1]
    text_ids = generated_ids[:, input_len:]
    decoded = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    extracted = extract_choice_letter(decoded)
    return extracted, decoded


def generate_answer_vllm(llm, sampling_params, processor, conversation, args):
    effective_use_audio_in_video = get_effective_use_audio_in_video(args)
    vllm_inputs = build_vllm_inputs(
        processor=processor,
        conversation=conversation,
        use_audio_in_video=effective_use_audio_in_video,
    )
    outputs = llm.generate([vllm_inputs], sampling_params=sampling_params)
    if not outputs or not outputs[0].outputs:
        return None, ""
    decoded = outputs[0].outputs[0].text
    extracted = extract_choice_letter(decoded)
    return extracted, decoded


def generate_answers_vllm_batch(
    llm, sampling_params, processor, batch_items, use_audio_in_video
):
    """
    Batch inference for vLLM.
    Returns a list (same length as batch_items), each element is:
    {
      "extracted_answer": str|None,
      "raw_output": str,
      "error": str|None
    }
    """
    results = [
        {"extracted_answer": None, "raw_output": "", "error": None}
        for _ in batch_items
    ]
    vllm_inputs = []
    input_to_item_index = []

    for item_index, item in enumerate(batch_items):
        try:
            conversation = build_conversation(
                media_paths=item["media_paths"],
                question=item["question"],
                choices=item["choices"],
                input_mode=item["input_mode"],
            )
            vllm_input = build_vllm_inputs(
                processor=processor,
                conversation=conversation,
                use_audio_in_video=use_audio_in_video,
            )
            vllm_inputs.append(vllm_input)
            input_to_item_index.append(item_index)
        except Exception as e:
            results[item_index]["error"] = str(e)

    if not vllm_inputs:
        return results

    try:
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
    except Exception as e:
        err_msg = str(e)
        for item_index in input_to_item_index:
            results[item_index]["error"] = err_msg
        return results

    if len(outputs) != len(vllm_inputs):
        print(
            f"Warning: vLLM batch output size mismatch: "
            f"{len(outputs)} outputs for {len(vllm_inputs)} inputs."
        )

    for out_pos, item_index in enumerate(input_to_item_index):
        if out_pos >= len(outputs):
            results[item_index]["error"] = "missing_output"
            continue
        output = outputs[out_pos]
        if not output.outputs:
            results[item_index]["raw_output"] = ""
            results[item_index]["extracted_answer"] = None
            continue
        decoded = output.outputs[0].text
        results[item_index]["raw_output"] = decoded
        results[item_index]["extracted_answer"] = extract_choice_letter(decoded)

    return results


def generate_answer(model, sampling_params, processor, media_paths, question, choices, args):
    conversation = build_conversation(
        media_paths=media_paths,
        question=question,
        choices=choices,
        input_mode=args.input_mode,
    )
    if args.use_vllm:
        return generate_answer_vllm(
            llm=model,
            sampling_params=sampling_params,
            processor=processor,
            conversation=conversation,
            args=args,
        )
    return generate_answer_transformers(
        model=model,
        processor=processor,
        conversation=conversation,
        args=args,
    )


def update_result_counters(
    *,
    qa_type,
    video_category,
    video_duration,
    is_correct,
    qa_type_count,
    qa_type_correct,
    video_cat_count,
    video_cat_correct,
    qa_duration_count,
    qa_duration_correct,
):
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


def test_all_questions(model, sampling_params, processor, args):
    qa_type_count = {}
    qa_type_correct = {}
    video_cat_count = {}
    video_cat_correct = {}

    data = load_json_data(args.json_file_path)
    if not data:
        print(f"Failed to load data from {args.json_file_path}. Exiting.")
        return

    video_categories = sorted(
        list({item.get("video_category") for item in data if item.get("video_category")})
    )
    qa_types = sorted(list({item.get("Type") for item in data if item.get("Type")}))

    for qa_type in qa_types:
        qa_type_count[qa_type] = 0
        qa_type_correct[qa_type] = 0
    for video_category in video_categories:
        video_cat_count[video_category] = 0
        video_cat_correct[video_category] = 0

    total_questions = len(data)
    correct_answers = 0
    failed = 0
    qa_duration_count = {"30s": 0, "60s": 0}
    qa_duration_correct = {"30s": 0, "60s": 0}
    item_results = []

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using video base directory: {args.video_base_dir}")
    print(f"Input mode: {args.input_mode}")
    use_batch_mode = args.use_vllm and args.enable_batch_mode
    if args.enable_batch_mode and not args.use_vllm:
        print("Warning: Batch mode is currently supported only with --use_vllm. Falling back to single-sample mode.")
    if use_batch_mode:
        print(f"vLLM batch mode enabled. Batch size: {args.batch_size}")

    pending_batch = []

    def append_item_result(
        item_meta,
        *,
        extracted_answer=None,
        raw_output="",
        is_correct=False,
        api_call_failed=False,
        skipped=False,
        reason=None,
    ):
        record = {
            "item_index": item_meta.get("idx"),
            "video_id": item_meta.get("video_id"),
            "question": item_meta.get("question"),
            "choices": item_meta.get("choices"),
            "correct_answer": item_meta.get("correct_answer"),
            "predicted_answer": extracted_answer,
            "is_correct": bool(is_correct),
            "api_call_failed": bool(api_call_failed),
            "skipped": bool(skipped),
            "reason": reason,
            "qa_type": item_meta.get("qa_type"),
            "video_category": item_meta.get("video_category"),
            "video_duration": item_meta.get("video_duration"),
            "input_mode": item_meta.get("input_mode"),
        }
        if args.save_raw_output:
            record["raw_output"] = raw_output
        item_results.append(record)

    def handle_prediction(item_meta, extracted_answer, raw_output):
        nonlocal correct_answers
        normalized_extracted = extracted_answer or extract_choice_letter(raw_output)
        is_correct = evaluate_answer(
            normalized_extracted or raw_output, item_meta["correct_answer"]
        )

        if args.verbose:
            print(
                f"\nItem {item_meta['idx']} | Video: {item_meta['video_id']} | "
                f"Pred: {normalized_extracted or raw_output!r} | "
                f"Gold: {item_meta['correct_answer']!r} | Correct: {is_correct}"
            )

        update_result_counters(
            qa_type=item_meta["qa_type"],
            video_category=item_meta["video_category"],
            video_duration=item_meta["video_duration"],
            is_correct=is_correct,
            qa_type_count=qa_type_count,
            qa_type_correct=qa_type_correct,
            video_cat_count=video_cat_count,
            video_cat_correct=video_cat_correct,
            qa_duration_count=qa_duration_count,
            qa_duration_correct=qa_duration_correct,
        )
        if is_correct:
            correct_answers += 1
        append_item_result(
            item_meta,
            extracted_answer=normalized_extracted,
            raw_output=raw_output,
            is_correct=is_correct,
            api_call_failed=False,
            skipped=False,
            reason=None,
        )

    def flush_batch():
        nonlocal failed, pending_batch
        if not pending_batch:
            return

        batch_results = generate_answers_vllm_batch(
            llm=model,
            sampling_params=sampling_params,
            processor=processor,
            batch_items=pending_batch,
            use_audio_in_video=get_effective_use_audio_in_video(args),
        )

        for item_meta, batch_result in zip(pending_batch, batch_results):
            err = batch_result.get("error")
            if err:
                print(
                    f"\nError processing video {item_meta['video_id']} "
                    f"(Index: {item_meta['idx']}): {err}"
                )
                failed += 1
                append_item_result(
                    item_meta,
                    extracted_answer=None,
                    raw_output=batch_result.get("raw_output", ""),
                    is_correct=False,
                    api_call_failed=True,
                    skipped=False,
                    reason=str(err),
                )
                continue
            handle_prediction(
                item_meta=item_meta,
                extracted_answer=batch_result.get("extracted_answer"),
                raw_output=batch_result.get("raw_output", ""),
            )

        if len(batch_results) < len(pending_batch):
            for item_meta in pending_batch[len(batch_results) :]:
                err = "missing_batch_output"
                print(
                    f"\nError processing video {item_meta['video_id']} "
                    f"(Index: {item_meta['idx']}): {err}"
                )
                failed += 1
                append_item_result(
                    item_meta,
                    extracted_answer=None,
                    raw_output="",
                    is_correct=False,
                    api_call_failed=True,
                    skipped=False,
                    reason=err,
                )

        pending_batch = []

    for idx, item in enumerate(tqdm.tqdm(data, desc="Evaluating Questions")):
        question = item.get("Question")
        choices = item.get("Choice")
        correct_answer = item.get("Answer")
        video_id = item.get("video_id")
        qa_type = item.get("Type")
        video_category = item.get("video_category")
        video_duration = item.get("video_duration")
        base_item_meta = {
            "idx": idx,
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "qa_type": qa_type,
            "video_category": video_category,
            "video_duration": video_duration,
            "input_mode": args.input_mode,
        }

        if not all(
            [
                question,
                choices,
                correct_answer,
                video_id,
                qa_type,
                video_category,
                video_duration,
            ]
        ):
            print(
                f"\nWarning: Skipping item due to missing fields. "
                f"Item index: {idx}, Video ID: {video_id or 'Unknown'}"
            )
            failed += 1
            append_item_result(
                base_item_meta,
                extracted_answer=None,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason="missing_fields",
            )
            continue

        try:
            if args.input_mode == "audio":
                audio_path = get_audio_path(video_id, args.video_base_dir)
                if not os.path.exists(audio_path):
                    print(
                        f"\nWarning: Audio file not found for ID {video_id} at path {audio_path}. Skipping."
                    )
                    failed += 1
                    append_item_result(
                        base_item_meta,
                        extracted_answer=None,
                        raw_output="",
                        is_correct=False,
                        api_call_failed=True,
                        skipped=True,
                        reason=f"audio_not_found:{audio_path}",
                    )
                    continue
                media_paths = {"audio_path": audio_path}
            elif args.input_mode == "all":
                video_path = get_video_path(video_id, args.video_base_dir)
                audio_path = get_audio_path(video_id, args.video_base_dir)
                missing = []
                if not os.path.exists(video_path):
                    missing.append(f"video={video_path}")
                if not os.path.exists(audio_path):
                    missing.append(f"audio={audio_path}")
                if missing:
                    print(
                        f"\nWarning: Missing media for ID {video_id}: {', '.join(missing)}. Skipping."
                    )
                    failed += 1
                    append_item_result(
                        base_item_meta,
                        extracted_answer=None,
                        raw_output="",
                        is_correct=False,
                        api_call_failed=True,
                        skipped=True,
                        reason=f"missing_media:{','.join(missing)}",
                    )
                    continue
                media_paths = {"video_path": video_path, "audio_path": audio_path}
            else:
                video_path = get_video_path(video_id, args.video_base_dir)
                if not os.path.exists(video_path):
                    print(
                        f"\nWarning: Video file not found for ID {video_id} at path {video_path}. Skipping."
                    )
                    failed += 1
                    append_item_result(
                        base_item_meta,
                        extracted_answer=None,
                        raw_output="",
                        is_correct=False,
                        api_call_failed=True,
                        skipped=True,
                        reason=f"video_not_found:{video_path}",
                    )
                    continue
                media_paths = {"video_path": video_path}
        except ValueError as e:
            print(
                f"\nError constructing media path: {e}. Skipping item for video ID {video_id}"
            )
            failed += 1
            append_item_result(
                base_item_meta,
                extracted_answer=None,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason=f"media_path_error:{e}",
            )
            continue

        item_meta = dict(base_item_meta)
        item_meta["media_paths"] = media_paths

        if use_batch_mode:
            pending_batch.append(item_meta)
            if len(pending_batch) >= args.batch_size:
                flush_batch()
            continue

        try:
            extracted_answer, raw_output = generate_answer(
                model=model,
                sampling_params=sampling_params,
                processor=processor,
                media_paths=media_paths,
                question=question,
                choices=choices,
                args=args,
            )
        except Exception as e:
            print(f"\nError processing video {video_id} (Index: {idx}): {e}")
            failed += 1
            append_item_result(
                item_meta,
                extracted_answer=None,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=False,
                reason=f"inference_error:{e}",
            )
            continue
        handle_prediction(
            item_meta=item_meta,
            extracted_answer=extracted_answer,
            raw_output=raw_output,
        )

    if use_batch_mode:
        flush_batch()

    print("\n--- Evaluation Summary ---")
    valid_questions = total_questions - failed
    if valid_questions > 0:
        print(
            f"Overall Accuracy: {correct_answers}/{valid_questions} = "
            f"{correct_answers / valid_questions:.2%}"
        )
    else:
        print("Overall Accuracy: 0/0 = N/A (No questions processed successfully)")
    print(f"(Total items: {total_questions}, Skipped/Failed items: {failed})")

    print("\n--- Accuracy by QA Type ---")
    for qa_type in qa_types:
        count = qa_type_count.get(qa_type, 0)
        correct = qa_type_correct.get(qa_type, 0)
        if count == 0:
            print(f"{qa_type}: 0/0 = N/A")
        else:
            print(f"{qa_type}: {correct}/{count} = {correct / count:.2%}")

    print("\n--- Accuracy by Video Category ---")
    for video_category in video_categories:
        count = video_cat_count.get(video_category, 0)
        correct = video_cat_correct.get(video_category, 0)
        if count == 0:
            print(f"{video_category}: 0/0 = N/A")
        else:
            print(f"{video_category}: {correct}/{count} = {correct / count:.2%}")

    print("\n--- Accuracy by Video Duration ---")
    for duration in ["30s", "60s"]:
        count = qa_duration_count.get(duration, 0)
        correct = qa_duration_correct.get(duration, 0)
        if count == 0:
            print(f"{duration} Duration: 0/0 = N/A")
        else:
            print(f"{duration} Duration: {correct}/{count} = {correct / count:.2%}")

    print(f"\nTotal items failed during processing: {failed}")
    item_results_path = args.item_results_path
    if not item_results_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        item_results_path = os.path.join(
            "runs", "qwen3_omni", f"item_results_{args.input_mode}_{ts}.jsonl"
        )
    written_path = save_item_results_jsonl(item_results, item_results_path)
    if written_path:
        print(f"Per-item results written to: {written_path}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-Omni on a Daily-Omni style video QA dataset."
    )
    parser.add_argument(
        "--video_base_dir",
        type=str,
        default="/data/Videos",
        help="Base directory containing video folders.",
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        default="/data/test_model/QA_all.json",
        help="Path to the JSON file containing QA pairs.",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="all",
        choices=["all", "visual", "audio"],
        help="Input modality for evaluation.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model name or local checkpoint path.",
    )
    parser.add_argument(
        "--processor_name_or_path",
        type=str,
        default=None,
        help="Processor name or path. Defaults to model_name_or_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device map for loading model (e.g. "auto", "cuda:0").',
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Model loading precision.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "none"],
        help='Attention implementation, use "none" for model default.',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2560,
        help="Maximum new tokens generated for each question.",
    )
    parser.add_argument(
        "--enable_talker",
        action="store_true",
        help="Keep talker enabled. Default disables talker for text-only evaluation.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM backend for inference (recommended by Qwen3-Omni README).",
    )
    parser.add_argument(
        "--enable_batch_mode",
        action="store_true",
        help="Enable batch inference (currently used for vLLM backend).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for vLLM batch inference.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=0,
        help="vLLM tensor_parallel_size. 0 means auto-detect from available CUDA devices.",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=1,
        help="vLLM max_num_seqs for offline generation.",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=32768,
        help="vLLM max_model_len.",
    )
    parser.add_argument(
        "--vllm_temperature",
        type=float,
        default=0.6,
        help="vLLM sampling temperature.",
    )
    parser.add_argument(
        "--vllm_top_p",
        type=float,
        default=0.95,
        help="vLLM top_p.",
    )
    parser.add_argument(
        "--vllm_top_k",
        type=int,
        default=20,
        help="vLLM top_k.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for vLLM backend.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample prediction details.",
    )
    parser.add_argument(
        "--item_results_path",
        type=str,
        default=None,
        help="Path to save per-item JSONL results (for Bootstrap CI).",
    )
    parser.add_argument(
        "--save_raw_output",
        action="store_true",
        default=True,
        help="Save raw model output text into per-item JSONL.",
    )

    args = parser.parse_args()
    if args.batch_size < 1:
        print("Error: --batch_size must be >= 1.")
        sys.exit(1)
    if args.processor_name_or_path is None:
        args.processor_name_or_path = args.model_name_or_path

    if process_mm_info is None:
        print("Error: qwen_omni_utils is required. Install it with `pip install qwen-omni-utils`.")
        sys.exit(1)

    dtype_map = {
        "auto": "auto",
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    load_dtype = dtype_map[args.precision]

    load_kwargs = {
        "device_map": args.device,
        "dtype": load_dtype,
    }
    if args.attn_implementation != "none":
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading processor: {args.processor_name_or_path}")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(args.processor_name_or_path)
    except Exception as e:
        print(f"Error loading processor: {e}")
        sys.exit(1)

    model = None
    sampling_params = None

    if args.use_vllm:
        # Keep consistent with Qwen3-Omni web_demo defaults for better compatibility.
        os.environ.setdefault("VLLM_USE_V1", "0")
        # vLLM may fail with "Cannot re-initialize CUDA in forked subprocess"
        # when worker multiprocessing method is fork. Force spawn by default.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            print(f"Error importing vLLM: {e}")
            print("Please install vLLM first, e.g. `pip install vllm==0.13.0`.")
            sys.exit(1)

        tp_size = args.vllm_tensor_parallel_size
        if tp_size <= 0:
            # Avoid touching torch.cuda before vLLM engine init.
            # Infer from CUDA_VISIBLE_DEVICES when possible; otherwise use 1.
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if cuda_visible_devices:
                tp_size = len(
                    [x for x in cuda_visible_devices.split(",") if x.strip() != ""]
                )
                tp_size = max(1, tp_size)
            else:
                tp_size = 1

        print(f"Loading model with vLLM: {args.model_name_or_path}")
        print(f"vLLM tensor_parallel_size: {tp_size}")
        print(f"vLLM gpu_memory_utilization: {args.vllm_gpu_memory_utilization}")
        print(f"vLLM max_model_len: {args.vllm_max_model_len}")
        print(f"vLLM max_num_seqs: {args.vllm_max_num_seqs}")

        try:
            model = LLM(
                model=args.model_name_or_path,
                trust_remote_code=True,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1},
                max_num_seqs=args.vllm_max_num_seqs,
                max_model_len=args.vllm_max_model_len,
                seed=args.seed,
            )
            sampling_params = SamplingParams(
                temperature=args.vllm_temperature,
                top_p=args.vllm_top_p,
                top_k=args.vllm_top_k,
                max_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"Error loading vLLM engine: {e}")
            sys.exit(1)
    else:
        print(f"Loading model with Transformers: {args.model_name_or_path}")
        print(f"Loading precision: {args.precision}")
        print(f"Attention implementation: {args.attn_implementation}")
        print(f"Device map: {args.device}")
        try:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                **load_kwargs,
            )
            if not args.enable_talker and hasattr(model, "disable_talker"):
                model.disable_talker()
        except Exception as e:
            print(f"Error loading Transformers model: {e}")
            sys.exit(1)

    test_all_questions(model, sampling_params, processor, args)
