import torch
# Updated imports for Qwen2.5 Omni
from transformers import Qwen2_5OmniProcessor
from transformers import Qwen2_5OmniForConditionalGeneration
# Assuming qwen_omni_utils contains process_mm_info
# Make sure this import works in your environment
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    print("Warning: qwen_omni_utils not found. Multimedia processing might fail.")
    process_mm_info = None

from typing import List, Dict, Any
import sys
# sys.path.append('./') # Keep if qwen_omni_utils is in the current dir
import argparse # Import argparse
import json
import tqdm
import os
import re
import time

# --- Removed Global Variables ---
# video_base_dir='/data/Videos'
# json_file_path='/data/test_model/QA_all.json'
# use_audio_in_video=False # Now handled by args
# -----------------------------

def load_json_data(file_path):
    """Loads JSON data from a file."""
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

# Modified to accept base_path as an argument
def get_video_path(video_id, base_path):
    """Constructs the video file path from a video ID."""
    if not base_path:
         raise ValueError("Video base path cannot be empty.")
    return os.path.join(base_path, video_id, f'{video_id}_video.mp4')


def get_audio_path(video_id, base_path):
    """Constructs the audio file path from a video ID."""
    if not base_path:
         raise ValueError("Video base path cannot be empty.")
    return os.path.join(base_path, video_id, f'{video_id}_audio.wav')


def get_effective_use_audio_in_video(_args):
    # The CLI now uses explicit modality selection only:
    # visual = video frames only, audio = audio only, all = both.
    return False


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
    video_content = None
    if input_mode != "audio":
        video_content = {"type": "video", "video": media_paths["video_path"]}
        if video_overrides:
            video_content.update(video_overrides)
    if input_mode == "audio":
        media_desc = "given audio"
        user_content = [{"type": "audio", "audio": media_paths["audio_path"]}]
    elif input_mode == "all":
        media_desc = "given video and audio together"
        user_content = [
            video_content,
            {"type": "audio", "audio": media_paths["audio_path"]},
        ]
    else:
        media_desc = "given video"
        user_content = [video_content]

    prompt = f"""
Your task is to accurately answer multiple-choice questions based on the {media_desc}.
Select the single most accurate answer from the given choices.
Question: {question}
Choices: {choices}
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
"""

    return [
        {"role": "system", "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."},
        ]},
        {"role": "user", "content": user_content + [{"type": "text", "text": prompt}]},
    ]


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


def build_vllm_inputs(processor, conversation, use_audio_in_video):
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )

    vllm_inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
    }
    if images is not None:
        vllm_inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        vllm_inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        vllm_inputs["multi_modal_data"]["audio"] = audios
    return vllm_inputs


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
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                inputs[key] = value.to(device=model.device, dtype=model.dtype)
            else:
                inputs[key] = value.to(device=model.device)

    gen_out = model.generate(
        **inputs,
        use_audio_in_video=effective_use_audio_in_video,
        return_audio=False,
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
    return extract_choice_letter(decoded_text), decoded_text


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
    decoded_text = outputs[0].outputs[0].text
    return extract_choice_letter(decoded_text), decoded_text


def is_engine_dead_error(exc):
    name = type(exc).__name__
    if name == "EngineDeadError":
        return True
    msg = str(exc).lower()
    return "enginedeaderror" in msg or "enginecore encountered an issue" in msg


def evaluate_answer(model_answer, correct_answer):
    extracted = extract_choice_letter(model_answer)
    if extracted is None:
        return False
    return extracted == correct_answer.strip().upper()


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


# Modified to use args for configuration
def test_all_questions(model, processor, args, sampling_params=None):
    """Tests all questions in the file using configuration from args."""
    qa_type_count = {}
    qa_type_correct = {}
    video_cat_count = {}
    video_cat_correct = {}

    # Load data using the path from args
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

    # --- Initial scan for categories and types ---
    for item in data:
        video_category = item.get('video_category')
        qa_type = item.get('Type')
        if video_category and video_category not in VIDEO_CAT:
            VIDEO_CAT.append(video_category)
        if qa_type and qa_type not in QA_TYPE:
            QA_TYPE.append(qa_type)

    VIDEO_CAT.sort()
    QA_TYPE.sort()

    for qa_type in QA_TYPE:
        qa_type_count[qa_type] = 0
        qa_type_correct[qa_type] = 0
    for video_category in VIDEO_CAT:
        video_cat_count[video_category] = 0
        video_cat_correct[video_category] = 0
    # ----------------------------------------------

    # data = data[800:810] # Keep for debugging if needed
    total_questions = len(data)
    correct_answers = 0
    failed = 0
    qa_duration_count = {"30s": 0, "60s": 0}
    qa_duration_correct = {"30s": 0, "60s": 0}

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using video base directory: {args.video_base_dir}")
    print(f"Input mode: {args.input_mode}")
    effective_input_mode = get_effective_input_mode(args)
    if effective_input_mode != args.input_mode:
        print(
            f"Effective input mode (backend-compatible): {effective_input_mode} "
            f"(from {args.input_mode})"
        )

    def append_item_result(
        item_meta,
        *,
        predicted_answer=None,
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
            "predicted_answer": predicted_answer,
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

        # Stricter check for required fields
        if not all([question, choices, correct_answer, video_id, qa_type, video_category, video_duration]):
            print(f"\nWarning: Skipping item due to missing fields. Item Index: {idx}, Video ID: {video_id or 'Unknown'}")
            failed += 1
            append_item_result(
                base_item_meta,
                predicted_answer=None,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason="missing_fields",
            )
            continue

        try:
            if effective_input_mode == "audio":
                audio_path = get_audio_path(video_id, args.video_base_dir)
                if not os.path.exists(audio_path):
                    print(f"\nWarning: Audio file not found for ID {video_id} at path {audio_path}. Skipping.")
                    failed += 1
                    append_item_result(
                        base_item_meta,
                        predicted_answer=None,
                        raw_output="",
                        is_correct=False,
                        api_call_failed=True,
                        skipped=True,
                        reason=f"audio_not_found:{audio_path}",
                    )
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
                    append_item_result(
                        base_item_meta,
                        predicted_answer=None,
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
                    print(f"\nWarning: Video file not found for ID {video_id} at path {video_path}. Skipping.")
                    failed += 1
                    append_item_result(
                        base_item_meta,
                        predicted_answer=None,
                        raw_output="",
                        is_correct=False,
                        api_call_failed=True,
                        skipped=True,
                        reason=f"video_not_found:{video_path}",
                    )
                    continue
                media_paths = {"video_path": video_path}
        except ValueError as e:
            print(f"\nError constructing media path: {e}. Skipping item for video ID {video_id}")
            failed += 1
            append_item_result(
                base_item_meta,
                predicted_answer=None,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason=f"media_path_error:{e}",
            )
            continue

        conversation = build_conversation(
            media_paths=media_paths,
            question=question,
            choices=choices,
            input_mode=effective_input_mode,
            video_overrides=get_video_sampling_overrides(args, video_duration),
        )
        model_answer = None # Initialize model_answer
        decoded_text = ""
        try:
            max_engine_restarts = args.vllm_engine_restart_retries if args.use_vllm else 0
            for attempt in range(max_engine_restarts + 1):
                try:
                    if args.use_vllm:
                        model_answer, decoded_text = generate_answer_vllm(
                            llm=model,
                            sampling_params=sampling_params,
                            processor=processor,
                            conversation=conversation,
                            args=args,
                        )
                    else:
                        model_answer, decoded_text = generate_answer_transformers(
                            model=model,
                            processor=processor,
                            conversation=conversation,
                            args=args,
                        )
                    break
                except Exception as inner_e:
                    should_restart = (
                        args.use_vllm
                        and is_engine_dead_error(inner_e)
                        and attempt < max_engine_restarts
                    )
                    if not should_restart:
                        raise
                    print(
                        f"\nWarning: EngineDeadError on video {video_id} (Index: {idx}). "
                        f"Restarting vLLM engine ({attempt + 1}/{max_engine_restarts})..."
                    )
                    model, sampling_params = load_vllm_backend(args)
            if model_answer is None:
                print(
                    f"\nWarning: Could not extract answer reliably from output "
                    f"for video {video_id}. Raw output: '{decoded_text}'"
                )

        except Exception as e:
            import traceback
            print(
                f"\nError processing video {video_id} (Index: {idx}): "
                f"{type(e).__name__}: {e!r}"
            )
            traceback.print_exc(limit=2)
            failed += 1
            append_item_result(
                base_item_meta,
                predicted_answer=None,
                raw_output=decoded_text,
                is_correct=False,
                api_call_failed=True,
                skipped=False,
                reason=f"inference_error:{type(e).__name__}:{e}",
            )
            continue # Skip to the next item

        normalized_model_answer = model_answer or extract_choice_letter(decoded_text)
        is_correct = evaluate_answer(normalized_model_answer, correct_answer)
        # Optional: Print intermediate results less frequently
        # if data.index(item) % 20 == 0:
        #     print(f"\nItem {data.index(item)} - Video: {video_id}")
        #     print(f"  Question: {question[:80]}...")
        #     print(f"  Model Answer Raw: '{model_answer}' (Extracted from: '{decoded_text}')") # Show extracted + raw
        #     print(f"  Correct Answer: {correct_answer}")
        #     print(f"  Result: {'Correct' if is_correct else 'Incorrect'}")


        # Update counts - ensure keys exist from the initial scan
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
            api_call_failed=False,
            skipped=False,
            reason=None,
        )

    # --- Results Reporting ---
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
        if count == 0:
            print(f"{qa_type}: 0/0 = N/A")
        else:
            print(f"{qa_type}: {correct}/{count} = {correct / count:.2%}")

    print('\n--- Accuracy by Video Category ---')
    for video_category in VIDEO_CAT:
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
        if count != 0:
            print(f"{duration} Duration: {correct}/{count} = {correct / count:.2%}")
        else:
            print(f"{duration} Duration: 0/0 = N/A")

    print(f"\nTotal items failed during processing: {failed}")
    item_results_path = args.item_results_path
    if not item_results_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        item_results_path = os.path.join(
            "runs", "qwen2_5_omni", f"item_results_{effective_input_mode}_{ts}.jsonl"
        )
    written_path = save_item_results_jsonl(item_results, item_results_path)
    if written_path:
        print(f"Per-item results written to: {written_path}")
    print("--- Evaluation Complete ---")


def _is_flash_attn_error(exc):
    msg = str(exc).lower()
    markers = [
        "flash_attn",
        "flash attention",
        "undefined symbol",
        "flash_attn_2_cuda",
        "c10_cuda_check_implementation",
    ]
    return any(marker in msg for marker in markers)


def _load_qwen_omni_model(model_name_or_path, device, dtype, attn_impl, enable_audio_output):
    # Keep compatibility across transformers versions that renamed torch_dtype -> dtype.
    def _try_load(chosen_attn):
        load_kwargs = {
            "device_map": device,
            "dtype": dtype,
            "enable_audio_output": enable_audio_output,
        }
        if chosen_attn is not None:
            load_kwargs["attn_implementation"] = chosen_attn
        try:
            return Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_name_or_path,
                **load_kwargs,
            )
        except TypeError as type_err:
            if "dtype" not in str(type_err):
                raise
            load_kwargs.pop("dtype", None)
            load_kwargs["torch_dtype"] = dtype
            return Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_name_or_path,
                **load_kwargs,
            )

    try:
        return _try_load(attn_impl), attn_impl
    except Exception as exc:
        if attn_impl != "flash_attention_2" or not _is_flash_attn_error(exc):
            raise

        print(f"FlashAttention load failed: {exc}")
        for fallback_impl in ["sdpa", "eager", None]:
            fallback_label = fallback_impl if fallback_impl is not None else "None (model default)"
            print(f"Retrying model load with attention implementation: {fallback_label}")
            try:
                return _try_load(fallback_impl), fallback_impl
            except Exception as fallback_exc:
                print(f"Fallback '{fallback_label}' failed: {fallback_exc}")
        raise


def load_vllm_backend(args):
    # Keep consistent with Qwen official demos for better compatibility.
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        print(f"Error importing vLLM: {e}")
        print("Please install vLLM first (see Qwen2.5-Omni README vLLM section).")
        raise

    tp_size = args.vllm_tensor_parallel_size
    if tp_size <= 0:
        # Avoid touching torch.cuda before vLLM engine init.
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible_devices:
            tp_size = len([x for x in cuda_visible_devices.split(",") if x.strip() != ""])
            tp_size = max(1, tp_size)
        else:
            tp_size = 1

    print(f"Loading model with vLLM: {args.model_name_or_path}")
    print(f"vLLM tensor_parallel_size: {tp_size}")
    print(f"vLLM gpu_memory_utilization: {args.vllm_gpu_memory_utilization}")
    print(f"vLLM max_model_len: {args.vllm_max_model_len}")
    print(f"vLLM max_num_seqs: {args.vllm_max_num_seqs}")
    print(
        "vLLM long-video sampling overrides: "
        f"fps={args.vllm_long_video_fps}, "
        f"min_frames={args.vllm_long_video_min_frames}, "
        f"max_frames={args.vllm_long_video_max_frames}"
    )
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
    return model, sampling_params


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni on a video QA dataset.")

    # --- Data Arguments ---
    parser.add_argument(
        '--video_base_dir',
        type=str,
        default='/data/Videos',
        help='Base directory containing video folders.'
    )
    parser.add_argument(
        '--json_file_path',
        type=str,
        default='/data/test_model/QA_all.json',
        help='Path to the JSON file containing QA pairs.'
    )

    # --- Processing Arguments ---

    parser.add_argument(
        '--input_mode',
        type=str,
        default='all',
        choices=['all', 'visual', 'audio'],
        help='Input modality for evaluation.'
    )

    # --- Model Loading Arguments ---
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='Qwen/Qwen2.5-Omni-7B',
        help='Hugging Face model name or path to load.'
    )
    parser.add_argument(
        '--processor_name_or_path',
        type=str,
        default=None,
        help='Hugging Face processor name or path. Defaults to model_name_or_path if not set.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device map for loading the model (e.g., "auto", "cuda:0").'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help='Precision for model loading (bf16 recommended for Ampere+ GPUs).'
    )
    parser.add_argument(
        '--attn_implementation',
        type=str,
        default='flash_attention_2',
        choices=['flash_attention_2', 'sdpa', 'eager', 'None'], # Added 'None' explicitly
        help='Attention implementation (set to "None" to disable or use default).'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=10,
        help='Maximum new tokens generated for each question.'
    )
    parser.add_argument(
        '--use_vllm',
        action='store_true',
        help='Use vLLM backend for inference (bypasses transformers flash-attn dependency).'
    )
    parser.add_argument(
        '--vllm_gpu_memory_utilization',
        type=float,
        default=0.95,
        help='vLLM gpu_memory_utilization.'
    )
    parser.add_argument(
        '--vllm_tensor_parallel_size',
        type=int,
        default=0,
        help='vLLM tensor_parallel_size. 0 means auto-detect from CUDA_VISIBLE_DEVICES.'
    )
    parser.add_argument(
        '--vllm_max_num_seqs',
        type=int,
        default=1,
        help='vLLM max_num_seqs for offline generation.'
    )
    parser.add_argument(
        '--vllm_max_model_len',
        type=int,
        default=32768,
        help='vLLM max_model_len.'
    )
    parser.add_argument(
        '--vllm_temperature',
        type=float,
        default=0.0,
        help='vLLM sampling temperature.'
    )
    parser.add_argument(
        '--vllm_top_p',
        type=float,
        default=1.0,
        help='vLLM top_p.'
    )
    parser.add_argument(
        '--vllm_top_k',
        type=int,
        default=-1,
        help='vLLM top_k.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Random seed for vLLM backend.'
    )
    parser.add_argument(
        '--vllm_engine_restart_retries',
        type=int,
        default=1,
        help='Retries to auto-restart vLLM engine after EngineDeadError.'
    )
    parser.add_argument(
        '--vllm_long_video_fps',
        type=float,
        default=1.0,
        help='Sampling fps override for 60s videos when using vLLM.'
    )
    parser.add_argument(
        '--vllm_long_video_min_frames',
        type=int,
        default=4,
        help='Minimum sampled frames override for 60s videos when using vLLM.'
    )
    parser.add_argument(
        '--vllm_long_video_max_frames',
        type=int,
        default=192,
        help='Maximum sampled frames override for 60s videos when using vLLM.'
    )
    parser.add_argument(
        '--disable_audio_output',
        action='store_false', # Flag to disable audio generation capability
        help='Disable audio output generation capability during transformers model loading.'
    )
    parser.add_argument(
        '--item_results_path',
        type=str,
        default=None,
        help='Path to save per-item JSONL results (for Bootstrap CI).'
    )
    parser.add_argument(
        '--save_raw_output',
        action='store_true',
        default=True,
        help='Save raw model output text into per-item JSONL.'
    )


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

    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    torch_dtype = dtype_map.get(args.precision, torch.bfloat16)

    attn_impl = args.attn_implementation if args.attn_implementation != "None" else None

    if process_mm_info is None:
        print("Error: qwen_omni_utils is required. Install it with `pip install qwen-omni-utils`.")
        sys.exit(1)

    print(f"Loading processor: {args.processor_name_or_path}...")
    try:
        processor = Qwen2_5OmniProcessor.from_pretrained(args.processor_name_or_path)
    except Exception as e:
        print(f"Error loading processor: {e}")
        sys.exit(1)

    model = None
    sampling_params = None
    try:
        if args.use_vllm:
            model, sampling_params = load_vllm_backend(args)
        else:
            print(f"Loading model with Transformers: {args.model_name_or_path}")
            print(f"Loading precision: {args.precision}")
            print(f"Attention implementation: {attn_impl}")
            print(f"Device map: {args.device}")
            print(f"Enable audio output: {not args.disable_audio_output}")
            model, resolved_attn_impl = _load_qwen_omni_model(
                model_name_or_path=args.model_name_or_path,
                device=args.device,
                dtype=torch_dtype,
                attn_impl=attn_impl,
                enable_audio_output=not args.disable_audio_output,
            )
            resolved_attn_label = resolved_attn_impl if resolved_attn_impl is not None else "None (model default)"
            print(f"Loaded model with attention implementation: {resolved_attn_label}")
    except Exception as e:
        print(f"Error loading backend: {e}")
        sys.exit(1)

    test_all_questions(model, processor, args, sampling_params=sampling_params)
