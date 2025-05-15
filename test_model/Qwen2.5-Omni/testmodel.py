import torch
# Updated imports for Qwen2.5 Omni
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
# Assuming qwen_omni_utils contains process_mm_info
# Make sure this import works in your environment
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    print("Warning: qwen_omni_utils not found. Multimedia processing might fail.")
    # Define a dummy function if needed for the script to run without it,
    # although actual processing will likely fail later.
    def process_mm_info(*args, **kwargs):
        print("Error: process_mm_info is not available.")
        # Return dummy values matching expected output structure if possible
        return None, None, None # Example: audios, images, videos

from typing import List, Dict, Any
import sys
# sys.path.append('./') # Keep if qwen_omni_utils is in the current dir
import argparse # Import argparse
import json
import tqdm
import os
import re

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

def evaluate_answer(model_answer, correct_answer):
    """Compares the model's answer with the correct answer."""
    # Handle potential None or empty string from the model
    if not model_answer:
        return False
    # Use a more robust check for a single capital letter answer
    return model_answer.strip().upper() == correct_answer.strip().upper()


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

    # data = data[800:810] # Keep for debugging if needed
    total_questions = len(data)
    correct_answers = 0
    failed = 0
    qa_duration_count={"30s":0, "60s":0}
    qa_duration_correct={"30s":0, "60s":0}

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using video base directory: {args.video_base_dir}")
    print(f"Use audio input: {args.use_audio_in_video}")


    for item in tqdm.tqdm(data, desc="Evaluating Questions"):
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')
        qa_type=item.get('Type')
        video_category=item.get('video_category')
        video_duration=item.get('video_duration')

        # Stricter check for required fields
        if not all([question, choices, correct_answer, video_id, qa_type, video_category, video_duration]):
            print(f"\nWarning: Skipping item due to missing fields. Item Index: {data.index(item)}, Video ID: {video_id or 'Unknown'}")
            failed += 1
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
        # Updated conversation structure for Omni
        conversation= [
            {"role": "system", "content": 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
            {"role": "user", "content": [
                {"type":"text","text":prompt},
                {"type":"video", "video": video_path}
            ]},
        ]
        model_answer = None # Initialize model_answer
        try:
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            # Use use_audio_in_video from args
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=args.use_audio_in_video)
            inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference
            # Pass use_audio_in_video from args
            # Generate only a few tokens, expecting just the letter
            gen_out = model.generate(
                **inputs,
                use_audio_in_video=args.use_audio_in_video,
                return_audio=False, # Already set by args.disable_audio_output during model load
                max_new_tokens=10,  # Limit generation length
                num_beams=1,        # Use greedy decoding
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id # Stop early if possible
            )
            # Decode generated IDs, excluding input IDs
            input_len = inputs['input_ids'].shape[1]
            text_ids = gen_out[:, input_len:]

            decoded_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Extract the answer (assuming it's the first letter after potential boilerplate)
            # This regex is specific to the observed output format "assistant\nA"
            # A more general approach might be needed if the format varies.
            match = re.search(r'assistant\s*([\s\S]*)$', decoded_text) # Find 'assistant' and capture everything after
            if match:
                 extracted = match.group(1).strip()
                 # Try to find the first capital letter A, B, C, or D
                 letter_match = re.match(r'\s*([A-D])', extracted, re.IGNORECASE)
                 if letter_match:
                     model_answer = letter_match.group(1).upper()
                 else:
                     print(f"\nWarning: Extracted text '{extracted}' for video {video_id} doesn't start with A, B, C, or D.")
                     model_answer = extracted # Keep the extracted text as is for inspection
            else:
                # If 'assistant' isn't found, maybe the model directly outputs the letter?
                 letter_match = re.match(r'\s*([A-D])', decoded_text, re.IGNORECASE)
                 if letter_match:
                     model_answer = letter_match.group(1).upper()
                 else:
                    print(f"\nWarning: Could not extract answer reliably from output for video {video_id}. Raw output: '{decoded_text}'")
                    model_answer = decoded_text # Keep raw output

        except Exception as e:
            import traceback
            print(f"\nError processing video {video_id} (Index: {data.index(item)}): {e}")
            # traceback.print_exc() # Uncomment for detailed traceback
            failed +=1
            continue # Skip to the next item

        is_correct = evaluate_answer(model_answer, correct_answer)
        # Optional: Print intermediate results less frequently
        # if data.index(item) % 20 == 0:
        #     print(f"\nItem {data.index(item)} - Video: {video_id}")
        #     print(f"  Question: {question[:80]}...")
        #     print(f"  Model Answer Raw: '{model_answer}' (Extracted from: '{decoded_text}')") # Show extracted + raw
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
        '--use_audio_in_video',
        action='store_true', # Flag to enable audio input processing
        help='Process audio input from the video along with visual frames.'
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
        '--disable_audio_output',
        action='store_false', # Flag to disable audio generation capability
        help='Disable audio output generation capability during model loading.'
    )


    args = parser.parse_args()

    if args.processor_name_or_path is None:
        args.processor_name_or_path = args.model_name_or_path

    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    torch_dtype = dtype_map.get(args.precision, torch.bfloat16)

    attn_impl = args.attn_implementation if args.attn_implementation != "None" else None

    print(f"Loading model: {args.model_name_or_path} with precision {args.precision}...")
    print(f"Attention implementation: {attn_impl}")
    print(f"Device map: {args.device}")
    print(f"Enable audio output: {not args.disable_audio_output}")

    try:
        model = Qwen2_5OmniModel.from_pretrained(
            args.model_name_or_path,
            device_map=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            enable_audio_output=not args.disable_audio_output, 
        )

        print(f"Loading processor: {args.processor_name_or_path}...")
        processor = Qwen2_5OmniProcessor.from_pretrained(args.processor_name_or_path)

    except Exception as e:
        print(f"Error loading model or processor: {e}")
        sys.exit(1)

    test_all_questions(model, processor, args)