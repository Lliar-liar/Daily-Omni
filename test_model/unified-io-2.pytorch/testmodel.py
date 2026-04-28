from uio2.model import UnifiedIOModel
import json
import tqdm
import os
from uio2.runner import TaskRunner
import argparse
import tensorflow as tf
import re
import sys
import time

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

def get_video_path(video_id, base_path):
    """Constructs the video file path from a video ID."""
    return os.path.join(base_path, video_id, f'{video_id}_video.mp4')

def ask_model(runner, video_path, question, choices):
    prompt = f"""
Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
Question: {question}
Choices: {choices}
"""
    return runner.avqa(video_path, prompt)

def evaluate_answer(model_answer, correct_answer):
    extracted = extract_choice_letter(model_answer)
    if extracted is None:
        return False
    return extracted == correct_answer.strip().upper()


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
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in sorted_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path

def test_all_questions(file_path, runner,args):  # Add runner as an argument
    """Tests all questions in the file."""
    data = load_json_data(file_path)
    if not data:
        return

    data = data[:100]  # Limit to 100 for testing - remove in full runs!
    total_questions = len(data)
    correct_answers = 0
    failed = 0
    item_results = []

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
            "input_mode": args.input_mode,
        }
        if args.save_raw_output:
            record["raw_output"] = raw_output
        item_results.append(record)

    for idx, item in enumerate(tqdm.tqdm(data, desc="Evaluating Questions")): # Add a description
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')
        item_meta = {
            "idx": idx,
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
        }

        if not all([question, choices, correct_answer, video_id]):
            print(f"Warning: Skipping item, missing required fields. Item: {item}")
            failed += 1
            append_item_result(
                item_meta,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason="missing_fields",
            )
            continue

        video_path = get_video_path(video_id,args.video_base_dir)
        if not os.path.exists(video_path):
            print(f"Warning: Video not found for {video_id} at {video_path}")
            failed += 1
            append_item_result(
                item_meta,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=True,
                reason=f"video_not_found:{video_path}",
            )
            continue

        try:  # Add a try-except block
            model_answer = ask_model(runner, video_path, question, choices)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            failed +=1 # increase failed counter.
            append_item_result(
                item_meta,
                raw_output="",
                is_correct=False,
                api_call_failed=True,
                skipped=False,
                reason=f"inference_error:{e}",
            )
            continue

        is_correct = evaluate_answer(model_answer, correct_answer)

        print(f"\nQuestion: {question}")
        print(f"  Model's Answer: {model_answer}")
        print(f"  Correct Answer: {correct_answer}")
        print(f"  Is Correct: {is_correct}")

        if is_correct:
            correct_answers += 1
        append_item_result(
            item_meta,
            raw_output=model_answer,
            is_correct=is_correct,
            api_call_failed=False,
            skipped=False,
            reason=None,
        )

    print(f"\nAccuracy: {correct_answers}/{total_questions} = {correct_answers / total_questions:.2%}")
    print(f"Failed: {failed}")
    item_results_path = args.item_results_path
    if not item_results_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        item_results_path = os.path.join(
            "runs", "unified_io_2", f"item_results_{args.input_mode}_{ts}.jsonl"
        )
    written_path = save_item_results_jsonl(item_results, item_results_path)
    if written_path:
        print(f"Per-item results written to: {written_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="allenai/uio2-large")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0).  Use -1 for CPU.') # Add GPU argument
    parser.add_argument(
        '--video_base_dir',
        type=str,
        default='Videos', # Added default from original global var
        help='Base directory containing video folders.'
    )
    parser.add_argument(
        '--json_file_path',
        type=str,
        default='qa.json', # Added default from original global var
        help='Path to the JSON file containing QA pairs.'
    )
    parser.add_argument(
        '--input_mode',
        type=str,
        default='all',
        choices=['all', 'visual', 'audio'],
        help='Input modality for evaluation.'
    )
    parser.add_argument(
        '--item_results_path',
        type=str,
        default=None,
        help='Path to save per-item JSONL results.'
    )
    parser.add_argument(
        '--save_raw_output',
        action='store_true',
        default=True,
        help='Save raw model output text into per-item JSONL.'
    )
    args = parser.parse_args()
    if args.input_mode != 'all':
        print("Error: unified-io-2.pytorch testmodel currently supports only --input_mode all.")
        sys.exit(1)

    # --- GPU Specification ---
    gpus = tf.config.list_physical_devices('GPU')
    if args.gpu >= 0 and gpus:  # If a specific GPU is requested AND GPUs are available
        try:
            tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[args.gpu], True) # Important for memory management
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Using GPU: {gpus[args.gpu]}, Logical GPUs: {logical_gpus}")
        except RuntimeError as e:
            print(e)
            print("Error setting GPU.  Falling back to CPU.")
            # Fallback to CPU is automatic if GPU setup fails.
    elif not gpus:
        print("No GPUs found.  Using CPU.")
    else:
      print("Using CPU.")


    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    # --- Model and Runner Initialization (AFTER GPU setup) ---
    from uio2.preprocessing import UnifiedIOPreprocessor
    preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor",
                                                         tokenizer="/home/unified_io2/tokenizer.model")
    model = UnifiedIOModel.from_pretrained(args.model).to('cuda')
    runner = TaskRunner(model, preprocessor)
    json_file_path = args.json_file_path
    test_all_questions(json_file_path, runner,args)
