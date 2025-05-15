import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# Assuming qwen_omni_utils contains process_vision_info
# Make sure this import works in your environment
try:
    from qwen_omni_utils import process_vision_info
except ImportError:
    print("Warning: qwen_omni_utils not found. Vision processing might fail.")
    # Define a dummy function if needed for the script to run without it,
    # although actual processing will likely fail later.
    def process_vision_info(*args, **kwargs):
        print("Error: process_vision_info is not available.")
        # Return dummy values matching expected output structure if possible
        return None, None, {} # Example: image_inputs, video_inputs, video_kwargs

from typing import List, Dict, Any
import sys
sys.path.append('./') # Keep if qwen_omni_utils is in the current dir
import argparse # Import argparse
import json
import tqdm
import os
import re


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

def evaluate_answer(model_answer, correct_answer):
    """Compares the model's answer with the correct answer."""
    # Handle potential None or empty string from the model
    if not model_answer:
        return False
    # Extract the first capital letter (A, B, C, D) if present
    match = re.match(r"\s*([A-D])", model_answer.strip().upper())
    if match:
        extracted_answer = match.group(1)
        return extracted_answer == correct_answer.upper()
    return False # If no valid answer format is found

# Modified to use args for configuration
def test_all_questions(model, processor, args):
    """Tests all questions in the file using configuration from args."""
    qa_type_count={}
    qa_type_correct={}
    video_cat_count={}
    video_cat_correct={}

    # Load data using the path from args
    data = load_json_data(args.json_file_path)
    if not data:
        print(f"Failed to load data from {args.json_file_path}. Exiting.")
        return

    total_questions = len(data)
    correct_answers = 0
    failed=0
    VIDEO_CAT=[]
    QA_TYPE=[]

    # --- Initial scan for categories and types ---
    for item in data:
        video_category=item.get('video_category')
        qa_type=item.get('Type')
        if video_category and video_category not in VIDEO_CAT:
            VIDEO_CAT.append(video_category)
        if qa_type and qa_type not in QA_TYPE:
            QA_TYPE.append(qa_type)

    VIDEO_CAT.sort()
    QA_TYPE.sort()

    for qa_type in QA_TYPE:
        qa_type_count[qa_type]=0
        qa_type_correct[qa_type]=0
    for video_category in VIDEO_CAT:
        video_cat_count[video_category]=0
        video_cat_correct[video_category]=0
    # ----------------------------------------------


    total_questions = len(data)
    correct_answers = 0
    failed = 0
    qa_duration_count={"30s":0, "60s":0}
    qa_duration_correct={"30s":0, "60s":0}

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using video base directory: {args.video_base_dir}")
    print(f"Sampling FPS: {args.fps}")
    # print(f"Use audio: {args.use_audio_in_video}") # Add back if audio logic is implemented

    for item in tqdm.tqdm(data, desc="Evaluating Questions"):
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')
        qa_type=item.get('Type')
        video_category=item.get('video_category')
        video_duration=item.get('video_duration')

        if not all([question, choices, correct_answer, video_id, qa_type, video_category, video_duration]):
            print(f"\nWarning: Skipping item due to missing fields. Item Index: {data.index(item)}, Video ID: {video_id or 'Unknown'}")
            failed += 1 # Count incomplete items as failures for tracking
            continue

        try:
            # Get video path using the base directory from args
            video_path = get_video_path(video_id, args.video_base_dir)
            if not os.path.exists(video_path):
                print(f"\nWarning: Video file not found for ID {video_id} at path {video_path}. Skipping.")
                failed += 1
                continue

        except ValueError as e:
             print(f"\nError constructing video path: {e}. Skipping item for video ID {video_id}")
             failed += 1
             continue

        prompt = f"""
Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.
Question: {question}
Choices: {choices}
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
"""
        # Renamed 'conversation' to 'messages' to match variable names later
        messages= [
            {"role": "user", "content": [
                {"type":"text","text":prompt},
                # Use fps from args
                {"type":"video","video":video_path,"fps":args.fps,"max_pixels": 360 * 420}
                # Add audio handling here if args.use_audio_in_video is True and implemented
            ]},
        ]
        try:
            # Apply chat template needs 'messages'
            text = processor.apply_chat_template(
                messages, # Use the correct variable name
                tokenize=False,
                add_generation_prompt=True
            )
            # Ensure process_vision_info is available and works
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            # Use fps from args here as well
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=args.fps, # Pass fps from args
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to(model.device) # Use model.device for flexibility

            # Inference
            # Added generation config for potentially better control/consistency
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10, # Restrict to a few tokens for A,B,C,D
                num_beams=1,       # Use greedy decoding for single letter
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id # Stop generation early
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # Decode only the first result (batch size is 1 here)
            model_answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0] # Get the string from the list
            # print(text) # Uncomment for debugging prompt
        except Exception as e:
            print(f"\nError processing video {video_id} (Index: {data.index(item)}): {e}")
            failed +=1 # increase failed counter.
            continue

        is_correct = evaluate_answer(model_answer, correct_answer)
        # Optional: Print intermediate results less frequently or based on a flag
        # if data.index(item) % 20 == 0: # Print every 20 items
        #     print(f"\nItem {data.index(item)} - Video: {video_id}")
        #     print(f"  Question: {question[:80]}...") # Truncate long questions
        #     print(f"  Model Answer Raw: '{model_answer}'")
        #     print(f"  Correct Answer: {correct_answer}")
        #     print(f"  Result: {'Correct' if is_correct else 'Incorrect'}")

        # Update counts - ensure keys exist from the initial scan
        if qa_type in qa_type_count:
            qa_type_count[qa_type]+=1
            if is_correct:
                 qa_type_correct[qa_type]+=1
        if video_category in video_cat_count:
             video_cat_count[video_category]+=1
             if is_correct:
                 video_cat_correct[video_category]+=1
        if video_duration in qa_duration_count:
             qa_duration_count[video_duration]+=1
             if is_correct:
                 qa_duration_correct[video_duration]+=1

        if is_correct:
            correct_answers += 1

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
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on a video QA dataset.")
    parser.add_argument(
        '--video_base_dir',
        type=str,
        default='Videos', # Default value if not provided
        help='Base directory containing video folders (e.g., /path/to/videos where each video is in /path/to/videos/video_id/video_id_video.mp4)'
    )
    parser.add_argument(
        '--json_file_path',
        type=str,
        default='qa.json', # Default value
        help='Path to the JSON file containing QA pairs.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2, # Default value
        help='Frames per second to sample from the video.'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='Qwen/Qwen2.5-VL-7B-Instruct',
        help='Hugging Face model name or path to load.'
    )
    parser.add_argument(
        '--processor_name_or_path',
        type=str,
        default=None, # Default to model name if not specified
        help='Hugging Face processor name or path. Defaults to model_name_or_path if not set.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device map for loading the model (e.g., "auto", "cuda:0"). "auto" distributes across available GPUs.'
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
        choices=['flash_attention_2', 'sdpa', 'eager', None],
        help='Attention implementation (requires compatible hardware/libraries).'
    )


    args = parser.parse_args()

    # Set processor path if not specified
    if args.processor_name_or_path is None:
        args.processor_name_or_path = args.model_name_or_path

    # Handle precision mapping
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    torch_dtype = dtype_map.get(args.precision, torch.bfloat16) # Default to bfloat16 if invalid choice

    # Handle optional attention implementation
    attn_impl = args.attn_implementation if args.attn_implementation != "None" else None


    # --- Model and Processor Loading ---
    print(f"Loading model: {args.model_name_or_path} with precision {args.precision}...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map=args.device,
        )

        print(f"Loading processor: {args.processor_name_or_path}...")
        processor = AutoProcessor.from_pretrained(args.processor_name_or_path)

    except Exception as e:
        print(f"Error loading model or processor: {e}")
        sys.exit(1) # Exit if loading fails

    # --- Run Evaluation ---
    # Pass the parsed args object to the evaluation function
    test_all_questions(model, processor, args)