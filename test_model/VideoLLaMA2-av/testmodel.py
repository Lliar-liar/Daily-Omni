import sys
# sys.path.append('./') # Keep if videollama2 is in the current dir or PYTHONPATH is set
try:
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init
except ImportError as e:
    print(f"Error importing videollama2: {e}")
    print("Please ensure the videollama2 library is installed and accessible.")
    print("You might need to install it or adjust your PYTHONPATH.")
    sys.exit(1)

import argparse
import json
import tqdm
import os
import re # Import re for potentially refining answer evaluation

# --- Removed Global Variables ---
# video_base_dir='/data/Videos'
# json_file_path='/data/test_model/QA_all.json'
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
    """Compares the model's answer with the correct answer (expecting single letter)."""
    # Handle potential None or empty string from the model
    if not model_answer:
        return False
    # Extract the first non-whitespace character and compare
    model_ans_processed = model_answer.strip()
    if not model_ans_processed:
        return False
    # More robust check for single letter A, B, C, D
    match = re.match(r"\s*([A-D])", model_ans_processed.upper())
    if match:
        extracted_answer = match.group(1)
        return extracted_answer == correct_answer.strip().upper()
    # Fallback to startswith for compatibility if needed, but less precise
    # return model_ans_processed.upper().startswith(correct_answer.strip().upper())
    print(f"Warning: Model answer '{model_ans_processed}' doesn't start with A, B, C, or D.")
    return False


# Modified to use args for configuration and removed file_path argument
def test_all_questions(model, processor, tokenizer, preprocess, args):
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

    print(f"\n--- Starting Evaluation ---")
    print(f"Dataset: {args.json_file_path}")
    print(f"Video Base Directory: {args.video_base_dir}")
    print(f"Model Path: {args.model_path}")
    print(f"Modal Type: {args.modal_type}")
    print(f"Total Questions: {total_questions}")
    print("-" * 30)


    for item_index, item in enumerate(tqdm.tqdm(data, desc="Evaluating Questions")):
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')
        qa_type=item.get('Type')
        video_category=item.get('video_category')
        video_duration=item.get('video_duration')

        # Stricter check for required fields
        if not all([question, choices, correct_answer, video_id, qa_type, video_category, video_duration]):
            print(f"\nWarning: Skipping item (Index: {item_index}) due to missing fields. Video ID: {video_id or 'Unknown'}")
            failed += 1
            continue

        try:
            # Get video path using the base directory from args
            video_path = get_video_path(video_id, args.video_base_dir)
            if not os.path.exists(video_path):
                print(f"\nWarning: Video file not found for ID {video_id} at path {video_path} (Index: {item_index}). Skipping.")
                failed += 1
                continue
        except ValueError as e:
             print(f"\nError constructing video path: {e}. Skipping item for video ID {video_id} (Index: {item_index})")
             failed += 1
             continue

        # --- Preprocessing based on modal_type ---
        try:
            if args.modal_type == "a":
                # Assuming preprocess expects only path for audio
                audio_video_tensor = preprocess(video_path)
            else: # 'v' or 'av'
                # Pass va=True only if modal_type is 'av'
                audio_video_tensor = preprocess(video_path, va=(args.modal_type == "av"))
        except Exception as e:
             print(f"\nError during preprocessing video {video_id} (Index: {item_index}): {e}")
             failed += 1
             continue
        # -----------------------------------------

        # --- Prompt Formatting ---
        prompt = f"""Your task is to accurately answer multiple-choice questions based on the given {'audio' if args.modal_type == 'a' else 'video'}.
Select the single most accurate answer from the given choices.
Question: {question}
Choices: {choices}
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text."""
        # -------------------------

        model_answer = None # Initialize
        try:
            # --- Inference ---
            model_answer = mm_infer(
                audio_video_tensor,
                prompt,
                model=model,
                processor=processor, # Pass processor if mm_infer needs it (check videollama2 docs)
                tokenizer=tokenizer,
                modal='audio' if args.modal_type == "a" else "video",
                do_sample=False,
                max_new_tokens=10 # Limit output length, expecting just a letter
            )
            # ----------------
        except Exception as e:
            print(f"\nError during inference for video {video_id} (Index: {item_index}): {e}")
            failed +=1
            continue # Skip to the next item

        is_correct = evaluate_answer(model_answer, correct_answer)

        # Optional: Print intermediate results less frequently
        # if item_index % 50 == 0:
        #      print(f"\n[Index {item_index}] Video: {video_id}")
        #      print(f"  Q: {question[:80]}...")
        #      print(f"  Model Ans: '{model_answer}' -> Correct: {is_correct} (Expected: {correct_answer})")

        # --- Update Counts ---
        if qa_type in qa_type_count: # Check if key exists (it should from initial scan)
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
        # ---------------------

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
    parser = argparse.ArgumentParser(description="Evaluate VideoLLaMA2 on a video QA dataset.")

    # --- Model/Path Arguments ---
    parser.add_argument(
        '--model-path',
        type=str,
        default='DAMO-NLP-SG/VideoLLaMA2.1-7B-AV', # Kept existing default
        help='Path or Hugging Face identifier for the VideoLLaMA2 model.'
        )
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

    # --- Processing Arguments ---
    parser.add_argument(
        '--modal-type',
        choices=["a", "v", "av"],
        default='av', # Kept existing default
        help='Modality to use: "a" (audio only), "v" (video only), "av" (audio-visual).'
        )

    # --- Optional: Add device argument if model_init supports it ---
    # parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda", "cpu").')

    args = parser.parse_args()

    # --- Initialization ---
    print("Initializing model...")
    try:
        # disable_torch_init() # Consider if this is necessary/desirable
        # Pass device if supported: model, processor, tokenizer = model_init(args.model_path, device=args.device)
        model, processor, tokenizer = model_init(args.model_path)
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error initializing model from {args.model_path}: {e}")
        sys.exit(1)
    # --------------------

    # --- Configure Model based on Modality ---
    print(f"Configuring model for modal type: {args.modal_type}")
    # Ensure model attributes exist before setting to None
    try:
        if args.modal_type == "a":
            if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
                 model.model.vision_tower = None
                 print("Disabled vision tower.")
            else:
                 print("Warning: Could not find model.model.vision_tower to disable.")
        elif args.modal_type == "v":
            if hasattr(model, 'model') and hasattr(model.model, 'audio_tower'):
                model.model.audio_tower = None
                print("Disabled audio tower.")
            else:
                print("Warning: Could not find model.model.audio_tower to disable.")
        elif args.modal_type == "av":
            print("Using both audio and video towers.")
            pass # Keep both active
        else:
             # This case should not be reached due to argparse choices, but good practice
             raise NotImplementedError(f"Modal type '{args.modal_type}' not implemented for model configuration.")

        preprocess_key = 'audio' if args.modal_type == "a" else "video"
        if preprocess_key in processor:
            preprocess = processor[preprocess_key]
            print(f"Using '{preprocess_key}' preprocessor.")
        else:
             raise ValueError(f"Preprocessor '{preprocess_key}' not found in the loaded processor dict.")

    except (AttributeError, NotImplementedError, ValueError) as e:
         print(f"Error configuring model or selecting preprocessor: {e}")
         sys.exit(1)

    test_all_questions(model, processor, tokenizer, preprocess, args)
    # --------------------