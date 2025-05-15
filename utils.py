import os
import base64
import time
import random
import google.generativeai as genai
import pandas as pd
import pathlib
import config # Import configuration

# --- Encoding ---

def encode_media(file_path):
    """Reads a media file (video or audio) and returns its base64 encoded content."""
    try:
        with open(file_path, "rb") as media_file:
            return base64.b64encode(media_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Media file not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"Error encoding media file {file_path}: {e}")
        return None

def read_media_bytes(file_path):
    """Reads a media file and returns its raw bytes."""
    try:
        return pathlib.Path(file_path).read_bytes()
    except FileNotFoundError:
        print(f"Error: Media file not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading media file bytes {file_path}: {e}")
        return None

# --- API Calls ---

def call_gemini_api(contents, api_key, model_name, system_prompt=None, max_retries=config.MAX_RETRIES, base_delay=config.BASE_DELAY):
    """
    Calls the Gemini API with retry logic and flexible configuration.

    Args:
        contents: The payload (list of parts) to send to the Gemini API.
        api_key (str): The API key to use for this call.
        model_name (str): The specific Gemini model to use.
        system_prompt (str, optional): System instruction for the model. Defaults to None.
        max_retries (int): Maximum number of retry attempts.
        base_delay (int): Base delay in seconds for exponential backoff.

    Returns:
        The API response object or None if retries fail or generation fails.
    """
    genai.configure(api_key=api_key, transport='rest')

    try:
        if system_prompt:
            model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
        else:
            model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing Gemini model '{model_name}': {e}")
        return None

    for attempt in range(max_retries):
        try:
            # Add a small delay before each attempt (except the first)
            if attempt > 0:
                # Basic delay + jitter
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"Waiting {delay:.2f} seconds before retry...")
                time.sleep(delay)
            
            # Consider adding a base delay even for the first attempt if needed
            # time.sleep(0.5) # Small base delay

            response = model.generate_content(contents)
            response.resolve() # Ensure the response is fully processed

            # Check for successful generation (finish_reason == 1 means STOP)
            # This check was present in audio_caption.py, applying generally
            if response and response.candidates:
                candidate = response.candidates[0]
                # FinishReason 1: STOP (Normal completion)
                # FinishReason 2: MAX_TOKENS
                # FinishReason 3: SAFETY
                # FinishReason 4: RECITATION
                # FinishReason 5: OTHER
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 1:
                    return response
                elif hasattr(candidate, 'finish_reason'):
                    print(f"Warning: Generation ended with reason: {candidate.finish_reason.name}")
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        print("  Safety Ratings:")
                        for rating in candidate.safety_ratings:
                            print(f"    Category: {rating.category.name}, Probability: {rating.probability.name}")
                    # Decide if we should retry for non-STOP reasons or return None/partial
                    # For now, we return the response even if not STOP=1, caller can check .text
                    return response # Or return None if only STOP=1 is acceptable
                else:
                     # If finish_reason is missing but we have candidates, return response
                     print("Warning: Candidate finish_reason attribute missing.")
                     return response
            elif response:
                 # Response object exists but no candidates (unusual)
                 print("Warning: API response received but no candidates found.")
                 # Check for prompt feedback if available
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"  Prompt Feedback: {response.prompt_feedback}")
                 return None # Indicate failure
            else:
                print("Warning: API call returned no response object.")
                # No need to retry if the API itself returned nothing.
                return None


        except Exception as e:
            error_str = str(e)
            # Check for rate limit errors (429) or resource exhausted errors
            if '429' in error_str or 'Resource has been exhausted' in error_str or 'service unavailable' in error_str.lower():
                # Exponential backoff with jitter for retries
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Received potentially transient error (e.g., 429, Resource Exhausted, Service Unavailable). Retrying in {delay:.2f} seconds (Attempt {attempt + 1}/{max_retries}). Details: {error_str}")
                time.sleep(delay)
            elif 'API key not valid' in error_str:
                print(f"Fatal Error: Invalid API Key. Please check configuration. Details: {error_str}")
                return None # No point retrying invalid key
            elif 'model' in error_str.lower() and ('not found' in error_str.lower() or 'does not support' in error_str.lower()):
                print(f"Fatal Error: Model '{model_name}' not found or incompatible. Please check configuration. Details: {error_str}")
                return None # No point retrying invalid model
            else:
                # For other types of errors, log and potentially retry or raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"An unexpected error occurred during API call (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f} seconds.")
                # raise # Option 1: Stop immediately for unexpected errors
                time.sleep(delay) # Option 2: Retry even for unexpected errors

    print(f"Error: Failed to get a valid response from Gemini API ({model_name}) after {max_retries} retries.")
    return None

# --- File and Directory Handling ---

def read_directory_list(csv_path, base_dir=config.BASE_DIR):
    """Reads a CSV file and returns a list of full directory paths."""
    dir_list = []
    try:
        df = pd.read_csv(csv_path)
        if "Directory" in df.columns:
            # Assumes "Directory" column contains full paths or relative paths from script location
            # For consistency, let's assume they are relative to base_dir or just IDs
            temp_list = df["Directory"].tolist()
            for item in temp_list:
                # If it looks like just an ID, join with base_dir
                if not os.path.isabs(item) and not os.path.sep in item:
                     dir_list.append(os.path.join(base_dir, str(item)))
                else:
                    # Assume it's already a usable path (absolute or relative)
                    # Consider resolving relative paths: os.path.abspath(item)
                    dir_list.append(item)

        elif "Video_id" in df.columns:
            video_ids = df["Video_id"].tolist()
            for video_id in video_ids:
                dir_list.append(os.path.join(base_dir, str(video_id)))
        else:
            print("Error: CSV file must contain either 'Directory' or 'Video_id' column.")
            return []
        return dir_list
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return []

def read_metadata(csv_path):
    """Reads specific metadata columns from CSV for QA generation."""
    metadata_list = []
    required_cols = ['content_parent_category', 'content_fine_category'] # Add more if needed
    try:
        df = pd.read_csv(csv_path)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Metadata columns missing in CSV '{csv_path}': {missing_cols}. Skipping metadata.")
            # Create dummy entries if needed downstream, or handle None
            if "Video_id" in df.columns:
                 return [{} for _ in range(len(df))] # Return list of empty dicts
            elif "Directory" in df.columns:
                 return [{} for _ in range(len(df))] # Return list of empty dicts
            else:
                 return None

        # Ensure Video_id or Directory exists to align metadata
        if "Video_id" not in df.columns and "Directory" not in df.columns:
             print("Error: Cannot align metadata without 'Video_id' or 'Directory' column.")
             return None

        # Extract required columns
        extracted_data = df[required_cols].to_dict('records')

        # Optionally add Video_id or Directory basename for alignment later
        if "Video_id" in df.columns:
            ids = df["Video_id"].tolist()
            for i, meta_dict in enumerate(extracted_data):
                meta_dict['video_id_for_alignment'] = ids[i]
                metadata_list.append(meta_dict)
        elif "Directory" in df.columns:
             dirs = df["Directory"].tolist()
             for i, meta_dict in enumerate(extracted_data):
                 # Use basename of directory as ID assuming it matches video ID
                 meta_dict['video_id_for_alignment'] = os.path.basename(dirs[i])
                 metadata_list.append(meta_dict)

        return metadata_list
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading metadata from CSV '{csv_path}': {e}")
        return None

def safe_read(file_path, encoding='utf-8'):
    """Safely reads a text file, returning None on error."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def safe_write(file_path, content, mode='w', encoding='utf-8'):
    """Safely writes content to a text file, returning True on success."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode, encoding=encoding, errors='replace') as f:
            f.write(content)
        return True
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred writing to file {file_path}: {e}")
        return False

def format_caption_with_timestamps(caption_path, segment_duration=config.SEGMENT_DURATION):
    """Reads lines from a caption file and prepends timestamps."""
    formatted_caption = ''
    cnt = 0
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            for line in f:
                start_time = cnt * segment_duration
                end_time = start_time + segment_duration
                timestamp = f"{start_time}-{end_time}s: "
                formatted_caption += timestamp + line.strip() + "\n" # Ensure one line break
                cnt += 1
        return formatted_caption.strip() # Remove trailing newline
    except FileNotFoundError:
        print(f"Warning: Caption file not found for timestamp formatting: {caption_path}")
        return "" # Return empty string if file not found
    except Exception as e:
        print(f"Error formatting captions with timestamps for {caption_path}: {e}")
        return ""
    
import glob # Needed for get_video_path

def get_video_path(video_id, base_path, segment_index=None, type='video', exact_match=False):
    """
    Constructs the path for a video or audio file (segment or full).
    If segment_index is None, tries to find the full video/audio file.
    If exact_match is True, requires the specific {video_id}_{type}.mp4/wav format.
    """
    dir_name = str(video_id) # Ensure it's a string
    extension = ".mp4" if type == 'video' else ".wav"

    if segment_index is not None:
        # Path for a segment
        return os.path.join(base_path, f"{dir_name}_{type}_{segment_index}{extension}")
    else:
        # Path for the full file
        specific_full_path = os.path.join(base_path, f"{dir_name}_{type}{extension}")
        if exact_match:
            if os.path.exists(specific_full_path):
                return specific_full_path
            else:
                # If exact match required but not found, search for alternatives
                 # Look for any file ending with the extension, not containing _{type}_
                pattern = os.path.join(base_path, f"*{extension}")
                possible_files = [f for f in glob.glob(pattern) if f"{type}_" not in os.path.basename(f)]
                if possible_files:
                    # Maybe sort by modification time or name? Return the first found for now.
                    print(f"Warning: Exact full file '{os.path.basename(specific_full_path)}' not found. Using alternative: '{os.path.basename(possible_files[0])}'")
                    return possible_files[0]
                else:
                    return None # Indicate not found
        else:
             # If exact match not required, check specific path first, then search
             if os.path.exists(specific_full_path):
                 return specific_full_path
             else:
                 # Look for any file ending with the extension, not containing _{type}_
                pattern = os.path.join(base_path, f"*{extension}")
                possible_files = [f for f in glob.glob(pattern) if f"{type}_" not in os.path.basename(f)]
                if possible_files:
                     print(f"Warning: Specific full file '{os.path.basename(specific_full_path)}' not found. Using first found: '{os.path.basename(possible_files[0])}'")
                     return possible_files[0] # Return the first one found
                else:
                     return None # Indicate not found

def log_gemini_failure_details(response):
    """Logs details from a failed Gemini API response object."""
    if response is None:
        print("  API call returned None object.")
        return

    if not response.text:
        print("  API response text is empty.")

    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        print(f"  Prompt Feedback: {response.prompt_feedback}")

    if hasattr(response, 'candidates') and response.candidates:
        for i, candidate in enumerate(response.candidates):
            print(f"  Candidate {i+1} Details:")
            if hasattr(candidate, 'finish_reason'):
                # Accessing enum value and name safely
                reason_val = getattr(candidate.finish_reason, 'value', 'N/A')
                reason_name = getattr(candidate.finish_reason, 'name', 'N/A')
                print(f"    Finish Reason: {reason_name} (Value: {reason_val})")
            else:
                print("    Finish Reason: N/A")

            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                 print("    Safety Ratings:")
                 for rating in candidate.safety_ratings:
                     category = getattr(rating.category, 'name', 'N/A')
                     probability = getattr(rating.probability, 'name', 'N/A')
                     print(f"      Category: {category}, Probability: {probability}")
            else:
                 print("    Safety Ratings: N/A")
    elif hasattr(response, 'candidates') and not response.candidates:
         print("  Response object has 'candidates' attribute, but it's empty.")
    else:
        print("  No 'candidates' attribute found in the response object.")