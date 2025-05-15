from uio2.model import UnifiedIOModel
import json
import tqdm
import os
from uio2.runner import TaskRunner
import argparse
import tensorflow as tf

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
    """Compares the model's answer with the correct answer."""
    # Handle potential None or empty string from the model
    if not model_answer:
        return False
    return model_answer.upper().startswith(correct_answer.upper())

def test_all_questions(file_path, runner,args):  # Add runner as an argument
    """Tests all questions in the file."""
    data = load_json_data(file_path)
    if not data:
        return

    data = data[:100]  # Limit to 100 for testing - remove in full runs!
    total_questions = len(data)
    correct_answers = 0
    failed = 0

    for item in tqdm.tqdm(data, desc="Evaluating Questions"): # Add a description
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')

        if not all([question, choices, correct_answer, video_id]):
            print(f"Warning: Skipping item, missing required fields. Item: {item}")
            continue

        video_path = get_video_path(video_id,args.video_base_dir)

        try:  # Add a try-except block
            model_answer = ask_model(runner, video_path, question, choices)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            failed +=1 # increase failed counter.
            continue

        is_correct = evaluate_answer(model_answer, correct_answer)

        print(f"\nQuestion: {question}")
        print(f"  Model's Answer: {model_answer}")
        print(f"  Correct Answer: {correct_answer}")
        print(f"  Is Correct: {is_correct}")

        if is_correct:
            correct_answers += 1

    print(f"\nAccuracy: {correct_answers}/{total_questions} = {correct_answers / total_questions:.2%}")
    print(f"Failed: {failed}")



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
    args = parser.parse_args()

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