# utils_tester.py

import json
import os
import time
import random
import base64
import subprocess
import cv2
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
import test_config as config # Import configuration

# --- File Handling ---

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
    except Exception as e:
        print(f"Error loading JSON data from '{file_path}': {e}")
        return None

def get_video_path(video_id, base_path=config.BASE_VIDEO_DIR):
    """Constructs the video file path from a video ID."""
    # Assumes video ID is the folder name and file is {video_id}_video.mp4
    return os.path.join(base_path, str(video_id), f'{str(video_id)}_video.mp4')

# --- Evaluation ---

def evaluate_answer(api_answer, correct_answer_char):
    """
    Compares API's answer with the correct answer character.
    Handles potential errors or None values from API.
    Returns True if correct, False otherwise.
    """
    if not api_answer or not isinstance(api_answer, str) or api_answer.startswith("error_"):
        return False
    # Check if the first non-whitespace character matches the correct answer
    return api_answer.strip().upper().startswith(correct_answer_char.strip().upper())

# --- Statistics Printing ---

def print_statistics(results_data, total_items_requested):
    """Calculates and prints detailed accuracy statistics."""
    print("\n--- Final Results ---")

    # Initialize containers
    stats = {
        'total_processed': 0,
        'overall_correct': 0,
        'api_failures': 0,
        'skipped': 0,
        'types': {},
        'categories': {},
        'durations': {}
    }
    type_counts = {}
    cat_counts = {}
    dur_counts = {}

    # Aggregate results
    for result in results_data:
        if result.get("skipped"):
            stats['skipped'] += 1
            # Count specific skip reasons if needed
            if result.get("reason") == "Video file not found":
                 # This is also counted as an API failure in the sense that the test couldn't run
                 stats['api_failures'] +=1
            continue # Don't process skipped items further for accuracy

        stats['total_processed'] += 1 # Count items that were attempted (not skipped)

        if result['api_call_failed']:
            stats['api_failures'] += 1
            # Don't count failed API calls in accuracy denominator

        # Only calculate accuracy for successfully processed items
        if not result['api_call_failed']:
            qa_type = result['qa_type']
            category = result['video_category']
            duration = result['video_duration']

            # Increment counts
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
            cat_counts[category] = cat_counts.get(category, 0) + 1
            dur_counts[duration] = dur_counts.get(duration, 0) + 1

            # Increment correct counts
            if result['is_correct']:
                stats['overall_correct'] += 1
                stats['types'][qa_type] = stats['types'].get(qa_type, 0) + 1
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
                stats['durations'][duration] = stats['durations'].get(duration, 0) + 1


    # --- Calculate and Print ---
    items_for_accuracy = stats['total_processed'] - stats['api_failures']

    if items_for_accuracy > 0:
        overall_acc = stats['overall_correct'] / items_for_accuracy
        print(f"Overall Accuracy: {stats['overall_correct']}/{items_for_accuracy} = {overall_acc:.2%}")
    else:
        print("Overall Accuracy: 0/0 = --- (No items successfully evaluated)")

    print(f"\nItems Requested: {total_items_requested}")
    print(f"Items Skipped (Missing fields/video): {stats['skipped']}")
    print(f"Items Attempted (Processed): {stats['total_processed']}")
    print(f"API Call Failures (Errors/Retries exceeded): {stats['api_failures']}")
    print(f"Items Evaluated for Accuracy: {items_for_accuracy}")


    print("\nAccuracy by QA Type:")
    for qa_type in sorted(type_counts.keys()):
        count = type_counts[qa_type]
        correct = stats['types'].get(qa_type, 0)
        acc = correct / count if count > 0 else 0
        print(f"  {qa_type}: {correct}/{count} = {acc:.2%}")

    print('\nAccuracy by Video Category:')
    for category in sorted(cat_counts.keys()):
        count = cat_counts[category]
        correct = stats['categories'].get(category, 0)
        acc = correct / count if count > 0 else 0
        print(f"  {category}: {correct}/{count} = {acc:.2%}")

    print("\nAccuracy by Video Duration:")
    for duration in sorted(dur_counts.keys()):
        count = dur_counts[duration]
        correct = stats['durations'].get(duration, 0)
        acc = correct / count if count > 0 else 0
        print(f"  {duration}: {correct}/{count} = {acc:.2%}")

    print("-" * 20)


# --- Video/Frame Processing ---

def encode_video_base64(video_path):
    """Encodes the entire video file to base64."""
    try:
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Video file not found for encoding: {video_path}")
        return None
    except Exception as e:
        print(f"Error encoding video {video_path}: {e}")
        return None

def encode_video_without_audio(video_path, ffmpeg_path=config.FFMPEG_PATH):
    """Encodes video without audio to base64 using ffmpeg."""
    temp_output_path = f"temp_no_audio_{os.path.basename(video_path)}_{random.randint(1000,9999)}.mp4"
    try:
        # Use '-y' to overwrite existing temp file if necessary
        command = [
            ffmpeg_path, '-i', video_path, '-an', '-vcodec', 'copy', '-y', temp_output_path,
            '-hide_banner', '-loglevel', 'error' # More succinct logging
        ]
        # print(f"Running ffmpeg command: {' '.join(command)}") # Debugging
        process = subprocess.run(command, check=True, capture_output=True, text=True)

        if not os.path.exists(temp_output_path):
             print(f"Error: ffmpeg command completed but output file '{temp_output_path}' not found.")
             print(f"ffmpeg stdout: {process.stdout}")
             print(f"ffmpeg stderr: {process.stderr}")
             return None

        encoded_video = encode_video_base64(temp_output_path)
        return encoded_video

    except FileNotFoundError:
         print(f"Error: ffmpeg command '{ffmpeg_path}' not found. Ensure ffmpeg is installed and in PATH or config_tester.py is updated.")
         return None
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg for {video_path}:")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error during audio removal for {video_path}: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception as e_rem:
                print(f"Warning: Could not remove temporary file {temp_output_path}: {e_rem}")

def extract_frames_base64(video_path, seconds_per_frame=config.SECONDS_PER_FRAME_GPT4O):
    """Extracts frames from video at specified intervals and encodes them."""
    base64Frames = []
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None # Indicate error

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            print(f"Warning: Invalid FPS ({fps}) for video {video_path}. Reading only first frame.")
            frames_to_skip = total_frames # Read only first frame effectively
        else:
            frames_to_skip = int(fps * seconds_per_frame)

        if frames_to_skip <= 0:
            frames_to_skip = 1 # Ensure progress

        curr_frame_idx = 0
        frames_extracted_count = 0
        while curr_frame_idx < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
            success, frame = video.read()
            if not success:
                 # Might happen at the very end, or if seeking fails
                 # print(f"Warning: Could not read frame at index {curr_frame_idx} from {video_path}")
                 curr_frame_idx += frames_to_skip # Still advance to avoid infinite loop
                 continue

            # Encode frame
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                print(f"Warning: Could not encode frame {curr_frame_idx} to JPG.")
                curr_frame_idx += frames_to_skip
                continue

            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            frames_extracted_count += 1
            curr_frame_idx += frames_to_skip

        # Ensure at least one frame is extracted if video is not empty
        if frames_extracted_count == 0 and total_frames > 0:
             video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Go to first frame
             success, frame = video.read()
             if success:
                  is_success, buffer = cv2.imencode(".jpg", frame)
                  if is_success:
                      base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                      frames_extracted_count += 1
                  else:
                       print("Warning: Could not encode the very first frame.")
             else:
                  print("Warning: Could not read the very first frame even on retry.")


        video.release()
        # print(f"Extracted {frames_extracted_count} frames from {video_path}")

        if not base64Frames:
             print(f"Error: No frames extracted from video {video_path}. Total frames: {total_frames}, FPS: {fps}")
             return None # Indicate error

        return base64Frames

    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        if 'video' in locals() and video.isOpened():
            video.release()
        return None

# --- Model-Specific API Call Functions ---

def _call_gemini_api(model_name, contents, system_prompt=None):
    """Internal helper to call Gemini API with retry logic."""
    genai.configure(api_key=config.GEMINI_API_KEY, transport='rest')

    # Convert string safety levels to HarmBlockThreshold enums
    safety_settings_enum = {}
    for category_str, threshold_str in config.GEMINI_SAFETY_SETTINGS.items():
        try:
            category_enum = getattr(HarmCategory, category_str)
            threshold_enum = getattr(HarmBlockThreshold, threshold_str)
            safety_settings_enum[category_enum] = threshold_enum
        except AttributeError:
            print(f"Warning: Invalid safety setting category or threshold: {category_str}/{threshold_str}")

    try:
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    except Exception as e:
        print(f"Error initializing Gemini model {model_name}: {e}")
        return "error_model_init"

    for attempt in range(config.MAX_RETRIES):
        try:
            # print(f"DEBUG: Calling Gemini ({model_name}) - Attempt {attempt+1}")
            response = model.generate_content(contents, safety_settings=safety_settings_enum)
            response.resolve() # Ensure completion

            # Basic check for text presence
            if response.text:
                return response.text.strip()
            else:
                 # Log detailed failure info if no text
                 print(f"Warning: Gemini response has no text (Attempt {attempt+1}).")
                 # Add more detailed logging like in the pipeline utils if needed
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     print(f"  Prompt Feedback: {response.prompt_feedback}")
                 if hasattr(response, 'candidates') and response.candidates:
                      reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')
                      print(f"  Finish Reason: {reason}")
                 # Continue to retry loop
                 # Consider returning specific error if blocked by safety, etc.
                 # For now, treat empty text as a retryable issue within limits
                 if attempt < config.MAX_RETRIES - 1:
                      delay = config.BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                      print(f"Retrying due to empty response in {delay:.2f}s...")
                      time.sleep(delay)
                 else:
                     print("Max retries reached with empty response.")
                     return "error_empty_response"


        except Exception as e:
            error_str = str(e).lower()
            delay = config.BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            if '429' in error_str or 'resource has been exhausted' in error_str or 'service unavailable' in error_str:
                print(f"Gemini API Error (429/Resource/Unavailable): {e}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{config.MAX_RETRIES})")
                time.sleep(delay)
            elif 'api key not valid' in error_str:
                 print(f"FATAL: Gemini API key not valid. Please check config. {e}")
                 return "error_invalid_key"
            elif 'model' in error_str and ('not found' in error_str or 'does not support' in error_str):
                 print(f"FATAL: Gemini model '{model_name}' not found or invalid. {e}")
                 return "error_invalid_model"
            else:
                print(f"Gemini API Error: {e}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{config.MAX_RETRIES})")
                time.sleep(delay)

    print(f"Gemini API Error: Max retries exceeded for model {model_name}.")
    return "error_max_retries"


def ask_gemini_av(question, choices, video_path):
    """Calls Gemini with full audio-visual video."""
    system_prompt = """
    Your task is to accurately answer multiple-choice questions based on the given video.
    Select the single most accurate answer from the given choices.
    Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
    """
    print(f"Processing (Gemini AV): {os.path.basename(video_path)}")
    encoded_video = encode_video_base64(video_path)
    if encoded_video is None:
        return "error_video_encoding"

    prompt = f'''Given the video, answer the question below. 
    Question: {question}
    Choices: {choices}'''
    contents = [
        {"mime_type": "video/mp4", "data": encoded_video},
        prompt
    ]
    return _call_gemini_api(config.GEMINI_AV_MODEL_NAME, contents, system_prompt)

def ask_gemini_visual(question, choices, video_path):
    """Calls Gemini with video only (no audio)."""
    system_prompt = f"""
    Your task is to accurately answer multiple-choice questions based on the given video.
    Select the single most accurate answer from the given choices. 
    Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
    """
    print(f"Processing (Gemini Visual): {os.path.basename(video_path)}")
    encoded_video_no_audio = encode_video_without_audio(video_path)
    if encoded_video_no_audio is None:
        return "error_audio_removal_or_encoding"

    prompt = f'''Given the video, answer the question below. 
    Question: {question}
    Choices: {choices}'''
    contents = [
        {"mime_type": "video/mp4", "data": encoded_video_no_audio},
        prompt
    ]
    return _call_gemini_api(config.GEMINI_VISUAL_MODEL_NAME, contents, system_prompt)

def _call_openai_compatible_api(client_config, model_name, messages, temperature=None, max_tokens=None):
    """Internal helper to call OpenAI-compatible APIs with retry logic."""
    client = OpenAI(api_key=client_config['api_key'], base_url=client_config['base_url'])

    request_params = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        request_params["temperature"] = temperature
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens
    else:
        # Default max_tokens for answer choices is small
        request_params["max_tokens"] = 10 # A, B, C, or D plus maybe short extra chars

    for attempt in range(config.MAX_RETRIES):
        try:
            # print(f"DEBUG: Calling {client_config.get('name', 'OpenAI-compatible')} ({model_name}) - Attempt {attempt+1}")
            completion = client.chat.completions.create(**request_params)
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                 return completion.choices[0].message.content.strip()
            else:
                 print(f"Warning: API response empty or malformed (Attempt {attempt+1}). Response: {completion}")
                 # Treat as retryable within limits
                 if attempt < config.MAX_RETRIES - 1:
                     delay = config.BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                     print(f"Retrying due to empty/malformed response in {delay:.2f}s...")
                     time.sleep(delay)
                 else:
                     print("Max retries reached with empty/malformed response.")
                     return "error_empty_response"

        except Exception as e:
            error_str = str(e).lower()
            delay = config.BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            if 'rate limit' in error_str or '429' in error_str:
                print(f"API Error (429/Rate Limit): {e}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{config.MAX_RETRIES})")
                time.sleep(delay)
            elif 'billing' in error_str or 'quota' in error_str:
                print(f"API Error (Billing/Quota): {e}. Stopping retries.")
                return "error_billing_quota"
            elif 'invalid api key' in error_str:
                 print(f"FATAL: Invalid API Key. Please check config. {e}")
                 return "error_invalid_key"
            elif 'authentication' in error_str: # Catch generic auth errors
                 print(f"FATAL: Authentication Error. Check API Key/Setup. {e}")
                 return "error_authentication"
            elif 'model' in error_str and ('not found' in error_str or 'does not exist' in error_str):
                 print(f"FATAL: Model '{model_name}' not found or invalid for this endpoint. {e}")
                 return "error_invalid_model"
            else:
                print(f"API Error: {e}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{config.MAX_RETRIES})")
                time.sleep(delay)

    print(f"API Error: Max retries exceeded for model {model_name}.")
    return "error_max_retries"

def ask_gpt4o_visual(question, choices, video_path):
    """Calls GPT-4o compatible API with video frames."""
    system_prompt = f"""
    Your task is to accurately answer multiple-choice questions based on the given video.
    Select the single most accurate answer from the given choices.
    Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
    """
    print(f"Processing (GPT4o Visual): {os.path.basename(video_path)}")
    base64Frames = extract_frames_base64(video_path)
    if base64Frames is None:
        return "error_frame_extraction"

    prompt_text = f'''Given the video, answer the question below.
Question: {question}
Choices: {choices}'''
    # Construct messages in OpenAI format
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                # Image URLs first
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
                    for frame_b64 in base64Frames
                ],
                # Then the text prompt
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    client_config = {'api_key': config.GPT4O_API_KEY, 'base_url': config.GPT4O_BASE_URL, 'name': 'GPT4o'}
    return _call_openai_compatible_api(client_config, config.GPT4O_MODEL_NAME, messages)


def ask_gpt4o_text(question, choices, video_path=None): # video_path not needed but kept for consistent signature
    """Calls GPT-4o compatible API with text only."""
    system_prompt = """
    Your task is to accurately answer multiple-choice questions.
    Select the single most accurate answer from the given choices. 
    Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
    """
    print(f"Processing (GPT4o Text): Question starting '{question[:30]}...'")
    prompt = f'''Answer the question about a video below. Give the most reasonable answer.
    Question: {question}
    Choices: {choices}'''
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    client_config = {'api_key': config.GPT4O_API_KEY, 'base_url': config.GPT4O_BASE_URL, 'name': 'GPT4o'}
    # Use default temperature and max_tokens from helper
    return _call_openai_compatible_api(client_config, config.GPT4O_MODEL_NAME, messages)

def ask_deepseek_text(question, choices, video_path=None): # video_path not needed
    """Calls DeepSeek compatible API with text only."""
    system_prompt = f"""
    Your task is to accurately answer multiple-choice questions.
    Select the single most accurate answer from the given choices. 
    Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
    """
    print(f"Processing (DeepSeek Text): Question starting '{question[:30]}...'")
    prompt = f'''Answer the question about a video below. Give the most reasonable answer.
    Question: {question}
    Choices: {choices}'''
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    client_config = {'api_key': config.DEEPSEEK_API_KEY, 'base_url': config.DEEPSEEK_BASE_URL, 'name': 'DeepSeek'}
    # Deepseek script had specific temp/tokens
    return _call_openai_compatible_api(client_config, config.DEEPSEEK_MODEL_NAME, messages, temperature=1.0, max_tokens=1000) 