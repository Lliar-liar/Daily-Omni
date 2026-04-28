import argparse
import json
import os
import re
import sys
import time

import tqdm
from transformers import AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


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


def extract_choice_letter(text):
    if not text:
        return None
    s = text.strip()
    if not s:
        return None

    first_char = s[0]
    if first_char in "ABCD":
        return first_char

    first_standalone = re.search(r"\b([ABCD])\b", s)
    if first_standalone:
        return first_standalone.group(1)

    return None


def evaluate_answer(model_answer, correct_answer):
    extracted = extract_choice_letter(model_answer)
    if extracted is None:
        return False
    return extracted == correct_answer.strip().upper()


def get_effective_input_mode(requested_mode):
    if requested_mode == "audio":
        raise ValueError("Qwen3-VL does not support audio-only evaluation.")
    if requested_mode == "all":
        print("Warning: Qwen3-VL is visual-only. Falling back from --input_mode all to visual.")
    return "visual"


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


def build_messages(video_path, question, choices, args):
    prompt = (
        "Your task is to accurately answer multiple-choice questions based on the given video.\n"
        "Select the single most accurate answer from the given choices.\n"
        f"Question: {question}\n"
        f"Choices: {choices}\n"
        "Your answer should be a capital letter representing your choice: A, B, C, or D. "
        "Don't generate any other text.\n"
    )

    video_item = {"type": "video", "video": video_path}
    if args.fps is not None:
        video_item["fps"] = args.fps

    return [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]


def prepare_inputs_for_vllm(messages, processor, args, return_video_metadata):
    """
    Follow Qwen3-VL README's vLLM offline inference format.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=return_video_metadata,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    mm_processor_kwargs = dict(video_kwargs or {})
    if args.do_sample_frames:
        mm_processor_kwargs["do_sample_frames"] = True
        if args.fps is not None:
            mm_processor_kwargs["fps"] = args.fps

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": mm_processor_kwargs,
    }


def should_retry_without_video_metadata(err_msg):
    if not err_msg:
        return False
    lowered = err_msg.lower()
    if "failed to apply qwen3vlprocessor" in lowered:
        return True
    if "qwen3vlprocessor" in lowered and "video" in lowered:
        return True
    return False


def should_retry_with_video_metadata(err_msg):
    if not err_msg:
        return False
    lowered = err_msg.lower()
    if "video metadata is required but not found in mm input" in lowered:
        return True
    if "video metadata is required" in lowered and "mm input" in lowered:
        return True
    return False


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


def evaluate_dataset(llm, sampling_params, processor, args):
    data = load_json_data(args.json_file_path)
    if not data:
        print(f"Failed to load data from {args.json_file_path}. Exiting.")
        return

    video_categories = sorted(
        list({item.get("video_category") for item in data if item.get("video_category")})
    )
    qa_types = sorted(list({item.get("Type") for item in data if item.get("Type")}))

    qa_type_count = {qa_type: 0 for qa_type in qa_types}
    qa_type_correct = {qa_type: 0 for qa_type in qa_types}
    video_cat_count = {cat: 0 for cat in video_categories}
    video_cat_correct = {cat: 0 for cat in video_categories}
    qa_duration_count = {"30s": 0, "60s": 0}
    qa_duration_correct = {"30s": 0, "60s": 0}

    total_questions = len(data)
    correct_answers = 0
    failed = 0
    item_results = []
    effective_input_mode = args.input_mode

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using video base directory: {args.video_base_dir}")
    print(f"Input mode: {effective_input_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"FPS in messages: {args.fps}")
    print(f"do_sample_frames: {args.do_sample_frames}")
    runtime_return_video_metadata = args.video_metadata_mode != "off"
    print(f"Initial return_video_metadata: {runtime_return_video_metadata}")

    pending_batch = []

    def append_item_result(
        item_meta,
        *,
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
            "predicted_answer": extract_choice_letter(raw_output),
            "is_correct": bool(is_correct),
            "api_call_failed": bool(api_call_failed),
            "skipped": bool(skipped),
            "reason": reason,
            "qa_type": item_meta.get("qa_type"),
            "video_category": item_meta.get("video_category"),
            "video_duration": item_meta.get("video_duration"),
            "input_mode": effective_input_mode,
        }
        if args.save_raw_output:
            record["raw_output"] = raw_output
        item_results.append(record)

    def handle_prediction(item_meta, raw_output):
        nonlocal correct_answers
        is_correct = evaluate_answer(raw_output, item_meta["correct_answer"])

        if args.verbose:
            print(
                f"\nItem {item_meta['idx']} | Video: {item_meta['video_id']} | "
                f"Pred: {extract_choice_letter(raw_output)!r} | "
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
            raw_output=raw_output,
            is_correct=is_correct,
            api_call_failed=False,
            skipped=False,
            reason=None,
        )

    def flush_batch():
        nonlocal failed, pending_batch, runtime_return_video_metadata
        if not pending_batch:
            return

        batch_inputs = [item["vllm_input"] for item in pending_batch]
        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        except Exception as e:
            err = str(e)
            fallback_metadata = None

            if runtime_return_video_metadata and should_retry_without_video_metadata(err):
                fallback_metadata = False
                print(
                    "\nWarning: vLLM failed with metadata-mode input; "
                    "retrying this batch with legacy video format."
                )
            elif (not runtime_return_video_metadata) and should_retry_with_video_metadata(err):
                fallback_metadata = True
                print(
                    "\nWarning: vLLM requires video metadata for current stack; "
                    "retrying this batch with metadata-mode input."
                )

            if fallback_metadata is None:
                for item in pending_batch:
                    print(
                        f"\nError processing video {item['video_id']} "
                        f"(Index: {item['idx']}): {err}"
                    )
                    failed += 1
                    append_item_result(
                        item,
                        raw_output="",
                        is_correct=False,
                        api_call_failed=True,
                        skipped=False,
                        reason=err,
                    )
                pending_batch = []
                return

            try:
                fallback_inputs = [
                    prepare_inputs_for_vllm(
                        item["messages"],
                        processor,
                        args,
                        return_video_metadata=fallback_metadata,
                    )
                    for item in pending_batch
                ]
                outputs = llm.generate(fallback_inputs, sampling_params=sampling_params)
                runtime_return_video_metadata = fallback_metadata
                print(
                    f"Info: switched runtime return_video_metadata="
                    f"{runtime_return_video_metadata} for next batches."
                )
            except Exception as e2:
                err = f"{err}\nRetry(with return_video_metadata={fallback_metadata}) failed: {e2}"
                for item in pending_batch:
                    print(
                        f"\nError processing video {item['video_id']} "
                        f"(Index: {item['idx']}): {err}"
                    )
                    failed += 1
                    append_item_result(
                        item,
                        raw_output="",
                        is_correct=False,
                        api_call_failed=True,
                        skipped=False,
                        reason=err,
                    )
                pending_batch = []
                return

        if len(outputs) != len(pending_batch):
            print(
                f"Warning: vLLM output size mismatch: "
                f"{len(outputs)} outputs for {len(pending_batch)} inputs."
            )

        for out_pos, item in enumerate(pending_batch):
            if out_pos >= len(outputs):
                print(
                    f"\nError processing video {item['video_id']} "
                    f"(Index: {item['idx']}): missing_output"
                )
                failed += 1
                append_item_result(
                    item,
                    raw_output="",
                    is_correct=False,
                    api_call_failed=True,
                    skipped=False,
                    reason="missing_output",
                )
                continue

            output = outputs[out_pos]
            if not output.outputs:
                print(
                    f"\nError processing video {item['video_id']} "
                    f"(Index: {item['idx']}): empty_output"
                )
                failed += 1
                append_item_result(
                    item,
                    raw_output="",
                    is_correct=False,
                    api_call_failed=True,
                    skipped=False,
                    reason="empty_output",
                )
                continue

            raw_output = output.outputs[0].text
            handle_prediction(item, raw_output)

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
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason="missing_fields",
            )
            continue

        try:
            video_path = get_video_path(video_id, args.video_base_dir)
            if not os.path.exists(video_path):
                print(
                    f"\nWarning: Video file not found for ID {video_id} at path {video_path}. Skipping."
                )
                failed += 1
                append_item_result(
                    base_item_meta,
                    raw_output="",
                    is_correct=False,
                    api_call_failed=True,
                    skipped=True,
                    reason=f"video_not_found:{video_path}",
                )
                continue
        except ValueError as e:
            print(f"\nError constructing video path: {e}. Skipping item for video ID {video_id}")
            failed += 1
            append_item_result(
                base_item_meta,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason=f"media_path_error:{e}",
            )
            continue

        try:
            messages = build_messages(video_path, question, choices, args)
            return_video_metadata = runtime_return_video_metadata
            vllm_input = prepare_inputs_for_vllm(
                messages,
                processor,
                args,
                return_video_metadata=return_video_metadata,
            )
        except Exception as e:
            print(f"\nError building vLLM input for video {video_id} (Index: {idx}): {e}")
            failed += 1
            append_item_result(
                base_item_meta,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=False,
                reason=f"build_input_error:{e}",
            )
            continue

        pending_batch.append(
            {
                **base_item_meta,
                "messages": messages,
                "vllm_input": vllm_input,
            }
        )

        if len(pending_batch) >= args.batch_size:
            flush_batch()

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
            "runs", "qwen3_vl", f"item_results_{effective_input_mode}_{ts}.jsonl"
        )
    written_path = save_item_results_jsonl(item_results, item_results_path)
    if written_path:
        print(f"Per-item results written to: {written_path}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on a Daily-Omni style video QA dataset with vLLM."
    )
    parser.add_argument(
        "--video_base_dir",
        type=str,
        default="Videos",
        help="Base directory containing video folders.",
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        default="qa.json",
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
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Hugging Face model name or local checkpoint path.",
    )
    parser.add_argument(
        "--processor_name_or_path",
        type=str,
        default=None,
        help="Processor name or path. Defaults to model_name_or_path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for vLLM inference.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum generation tokens per sample.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS written into message video config.",
    )
    parser.add_argument(
        "--do_sample_frames",
        action="store_true",
        help="Enable frame sampling controls in mm_processor_kwargs.",
    )
    parser.add_argument(
        "--video_metadata_mode",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            'Video metadata mode for qwen_vl_utils process_vision_info. '
            '"on": always use return_video_metadata=True (README style). '
            '"off": use legacy format. '
            '"auto": try "on", and retry failed batches with "off" when Qwen3VLProcessor compatibility errors occur.'
        ),
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
        help="vLLM tensor_parallel_size. 0 means auto-detect from CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=8,
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
        default=0.0,
        help="vLLM sampling temperature.",
    )
    parser.add_argument(
        "--vllm_top_p",
        type=float,
        default=1.0,
        help="vLLM top_p.",
    )
    parser.add_argument(
        "--vllm_top_k",
        type=int,
        default=-1,
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
        help="Path to save per-item JSONL results.",
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
    try:
        args.input_mode = get_effective_input_mode(args.input_mode)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if args.processor_name_or_path is None:
        args.processor_name_or_path = args.model_name_or_path

    if process_vision_info is None:
        print("Error: qwen_vl_utils is required. Install with `pip install qwen-vl-utils==0.0.14`.")
        sys.exit(1)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        print(f"Error importing vLLM: {e}")
        print("Please install vLLM first, e.g. `pip install -U vllm`.")
        sys.exit(1)

    tp_size = args.vllm_tensor_parallel_size
    if tp_size <= 0:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible_devices:
            tp_size = len([x for x in cuda_visible_devices.split(",") if x.strip()])
            tp_size = max(1, tp_size)
        else:
            tp_size = 1

    print(f"Loading processor: {args.processor_name_or_path}")
    try:
        processor = AutoProcessor.from_pretrained(args.processor_name_or_path)
    except Exception as e:
        print(f"Error loading processor: {e}")
        sys.exit(1)

    print(f"Loading vLLM model: {args.model_name_or_path}")
    print(f"vLLM tensor_parallel_size: {tp_size}")
    print(f"vLLM gpu_memory_utilization: {args.vllm_gpu_memory_utilization}")
    print(f"vLLM max_model_len: {args.vllm_max_model_len}")
    print(f"vLLM max_num_seqs: {args.vllm_max_num_seqs}")

    try:
        llm = LLM(
            model=args.model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 1},
            max_num_seqs=args.vllm_max_num_seqs,
            max_model_len=args.vllm_max_model_len,
            seed=args.seed,
        )
        sampling_params = SamplingParams(
            temperature=args.vllm_temperature,
            top_p=args.vllm_top_p,
            top_k=args.vllm_top_k,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[],
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        sys.exit(1)

    evaluate_dataset(llm, sampling_params, processor, args)
