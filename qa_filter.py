import json
import os
import time
import random
import base64
import multiprocessing # Keep for potential parallel filtering later, though sequential for now
from openai import OpenAI
import tqdm
import config # Import configuration
import utils  # Import utils for get_video_path

def load_json_data(file_path, max_retries=10):
    """Loads JSON data from a file."""
    # This function might not be needed if we pass the list directly
    for i in range(max_retries):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'")
            # Don't retry indefinitely here if called from pipeline
            return None
            # time.sleep(1)
            # continue
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{file_path}'")
             # Don't retry indefinitely here if called from pipeline
            return None
            # time.sleep(1)
            # continue
    return None

# Generic function to call text-only models for filtering
def ask_text_model_for_filter(question, choices, model_name, api_key, base_url, system_prompt=config.QA_FILTER_SYSTEM_PROMPT, max_retries=config.MAX_RETRIES, base_delay=config.BASE_DELAY):
    """Asks a text-based model to guess the answer based on question/choices only."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f'''Answer the question about a video below. Give the most reasonable answer.
Question: {question}
Choices: {choices}'''

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=10 # Only need the letter A, B, C, or D
            )
            answer = response.choices[0].message.content.strip()[0]

            # Basic validation: Ensure it's a single uppercase letter A-D
            if len(answer) == 1 and answer in ['A', 'B', 'C', 'D']:
                return answer
            else:
                print(f"Warning: Model {model_name} returned invalid format: '{answer}'. Treating as error.")
                 # Fall through to retry or return error

        except Exception as e:
            error_str = str(e).lower()
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                print(f"Filter Model {model_name}: 429 Error. Retrying in {delay:.2f}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            # Add specific checks for invalid API key or model not found if needed
            elif 'invalid api key' in error_str:
                print(f"FATAL: Invalid API Key for {model_name} ({base_url}). Aborting filter for this item.")
                return "error_fatal" # Special code to stop retrying
            else:
                print(f"Filter Model {model_name}: HTTP Error: {e}. Retrying in {delay:.2f}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)

    print(f"Filter Model {model_name}: Max retries exceeded for question: {question}")
    return "error"

def evaluate_filter_answer(answer1, answer2, correct_answer):
    """Checks if both model answers match the correct answer."""
    # Ensure answers are valid before comparison
    if answer1 in ['error', 'error_fatal'] or answer2 in ['error', 'error_fatal']:
        return False # Treat API errors as failure

    # Get the first character and uppercase it (handle potential extra chars robustly)
    first_char1 = answer1[0].upper() if answer1 else '?'
    first_char2 = answer2[0].upper() if answer2 else '?'
    first_char_correct = correct_answer[0].upper() if correct_answer else '!'

    # Check if all three valid chars (A, B, C, D) are the same
    valid_chars = ['A', 'B', 'C', 'D']
    if first_char1 in valid_chars and first_char2 in valid_chars and first_char_correct in valid_chars:
        return first_char1 == first_char2 == first_char_correct
    else:
        print(f"Warning: Invalid characters in filter evaluation: M1='{first_char1}', M2='{first_char2}', Correct='{first_char_correct}'")
        return False

def filter_qa_list(qa_list, base_video_dir):
    """
    Filters a list of QA dictionaries based on text-only model agreement.

    Args:
        qa_list (list): A list of QA dictionaries.
        base_video_dir (str): The base directory where videos are stored (used by get_video_path).

    Returns:
        list: A new list containing only the QA dictionaries that passed the filter.
    """
    if not qa_list:
        return []

    filtered_qas = []
    passed_count = 0
    failed_count = 0
    error_count = 0

    # Use tqdm for progress bar
    for item in tqdm.tqdm(qa_list, desc="Filtering QAs"):
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')

        if not all([question, choices, correct_answer, video_id]):
            print(f"Warning: Skipping item in filter, missing required fields. Item: {item}")
            failed_count +=1
            continue


        model1_answer = ask_text_model_for_filter(
            question, choices,
            config.FILTER_MODEL_1_NAME,
            config.FILTER_MODEL_1_API_KEY,
            config.FILTER_MODEL_1_BASE_URL
        )

        model2_answer = ask_text_model_for_filter(
            question, choices,
            config.FILTER_MODEL_2_NAME,
            config.FILTER_MODEL_2_API_KEY,
            config.FILTER_MODEL_2_BASE_URL
        )

        # Handle fatal API errors (like invalid key)
        if model1_answer == "error_fatal" or model2_answer == "error_fatal":
             print(f"FATAL API error encountered for QA Item (ID: {video_id}, Q: {question[:30]}...). Skipping this item.")
             error_count += 1
             continue # Skip this item entirely

        if model1_answer == "error" or model2_answer == "error":
            print(f"API error encountered for QA Item (ID: {video_id}, Q: {question[:30]}...). Filter Check Failed.")
            error_count += 1
            # Failed API calls mean the check fails
            is_correct = False
        else:
             # Evaluate if both models agreed with the correct answer
            is_correct = evaluate_filter_answer(model1_answer, model2_answer, correct_answer)

        # Print intermediate results (optional, can be noisy)
        # print(f"\nFilter Check for Video ID: {video_id}")
        # print(f"  Question: {question}")
        # print(f"  Choices: {choices}")
        # print(f"  Model 1 ({config.FILTER_MODEL_1_NAME}): {model1_answer}")
        # print(f"  Model 2 ({config.FILTER_MODEL_2_NAME}): {model2_answer}")
        # print(f"  Correct Answer: {correct_answer}")
        # print(f"  Passed Filter: {is_correct}")

        if is_correct==False:
            filtered_qas.append(item)
            passed_count += 1
        else:
            failed_count += 1 # Includes items skipped due to missing fields/paths

    print("-" * 30)
    print("QA Filtering Summary:")
    print(f"  Total items processed: {len(qa_list)}")
    print(f"  Passed filter: {passed_count}")
    print(f"  Failed filter (incl. missing data): {failed_count}")
    print(f"  API errors during filter: {error_count}")
    print("-" * 30)

    return filtered_qas

# Note: No __main__ block needed here, as it's imported by run_pipeline.py