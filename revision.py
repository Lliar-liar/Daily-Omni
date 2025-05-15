import os
import utils
import config

def revise_video_consistency(video_folder_path):
    """
    Revises video captions for consistency across segments using the full video.
    Reads 'video_detail_captions.txt', writes 'video_consistent_captions.txt'.
    """
    print(f"\n--- Starting Video Consistency Revision for: {video_folder_path} ---")
    dir_name = os.path.basename(video_folder_path)

    # Path to the *full* video file (assuming it exists)
    full_video_path = os.path.join(video_folder_path, f"{dir_name}_video.mp4")
    if not os.path.exists(full_video_path):
        # Try finding any .mp4 file if the specific name doesn't exist
        found_video = None
        for f in os.listdir(video_folder_path):
             if f.endswith(".mp4") and not f.startswith(f"{dir_name}_video_"): # Avoid segments
                  found_video = os.path.join(video_folder_path, f)
                  break
        if found_video:
             full_video_path = found_video
             print(f"  Using video file: {os.path.basename(full_video_path)}")
        else:
            print(f"Error: Full video file ('{dir_name}_video.mp4' or similar) not found in {video_folder_path}. Skipping video revision.")
            return False

    # Path to the input captions
    input_caption_path = os.path.join(video_folder_path, "video_detail_captions.txt")
    if not os.path.exists(input_caption_path):
        print(f"Error: Input caption file 'video_detail_captions.txt' not found in {video_folder_path}. Skipping video revision.")
        return False

    # Path for the output revised captions
    output_caption_path = os.path.join(video_folder_path, "video_consistent_captions.txt")

    # Read and format the input captions with timestamps
    original_captions_formatted = utils.format_caption_with_timestamps(input_caption_path)
    if not original_captions_formatted:
        print(f"Error: Could not read or format input captions from {input_caption_path}. Skipping video revision.")
        return False

    print("Input captions loaded. Encoding full video and calling revision API...")

    # Encode the full video
    # encoded_video = utils.encode_media(full_video_path)
    # if encoded_video is None:
    #     print(f"Error encoding full video file: {full_video_path}. Skipping video revision.")
    #     return False
    # contents = [
    #     {"mime_type": "video/mp4", "data": encoded_video},
    #     f"Improve the consistency of the captions.\nVideo caption:\n{original_captions_formatted}" # User prompt
    # ]
    
    # Alternative: Use file bytes (check model limits for full video length)
    # May need Gemini File API for longer videos
    video_bytes = utils.read_media_bytes(full_video_path)
    if video_bytes is None:
         print(f"Error reading full video file: {full_video_path}. Skipping video revision.")
         return False
    contents = [
        {"mime_type": "video/mp4", "data": video_bytes},
        f"Improve the consistency of the captions.\nVideo caption:\n{original_captions_formatted}" # User prompt
    ]


    response = utils.call_gemini_api(
        contents=contents,
        api_key=config.GEMINI_API_KEY_REVISION, # Or default key
        model_name=config.GEMINI_REVISION_MODEL, # Use a model suitable for revision
        system_prompt=config.VIDEO_REVISION_SYSTEM_PROMPT
    )

    if response and response.text:
        revised_caption = response.text.strip()
        print(f"  Generated Revised Caption: {revised_caption[:150]}...")
        if utils.safe_write(output_caption_path, revised_caption + "\n", mode='w'):
            print(f"--- Video Consistency Revision Complete for: {video_folder_path}. Saved to {os.path.basename(output_caption_path)}. ---")
            return True
        else:
            print(f"  Error: Failed to write revised video captions to {output_caption_path}.")
            return False
    else:
        print("  Warning: Failed to generate revised video captions.")
        return False


def revise_audio_captions(audio_folder_path):
    """
    Revises audio captions based on consistency and visual context.
    Reads 'audio_detail_captions.txt' and 'video_consistent_captions.txt'.
    Writes 'audio_revised_captions.txt'.
    Uses the specific Gemini key and potentially 'thinking' model from original script.
    """
    print(f"\n--- Starting Audio Revision for: {audio_folder_path} ---")
    dir_name = os.path.basename(audio_folder_path)

    # Input file paths
    audio_caption_path = os.path.join(audio_folder_path, "audio_detail_captions.txt")
    video_caption_path = os.path.join(audio_folder_path, "video_consistent_captions.txt") # Use the consistent video captions
    output_caption_path = os.path.join(audio_folder_path, 'audio_revised_captions.txt')

    # Check if input files exist
    if not os.path.exists(audio_caption_path):
        print(f"Error: Input audio caption file 'audio_detail_captions.txt' not found. Skipping audio revision.")
        return False
    if not os.path.exists(video_caption_path):
        print(f"Error: Input video caption file 'video_consistent_captions.txt' not found. Skipping audio revision.")
        return False

    # Read and format input captions
    audio_captions_formatted = utils.format_caption_with_timestamps(audio_caption_path)
    # Video captions might already be timestamped by video revision, or need formatting if not
    # Assuming video_consistent_captions.txt *is* timestamped correctly by revise_video_consistency
    video_captions_content = utils.safe_read(video_caption_path)

    if not audio_captions_formatted:
        print(f"Error: Could not read or format input audio captions from {audio_caption_path}. Skipping.")
        return False
    if video_captions_content is None:
        print(f"Error: Could not read input video captions from {video_caption_path}. Skipping.")
        return False

    print("Input captions loaded. Calling revision API...")

    # Prepare prompt for Gemini 'thinking' model (or standard revision model)
    # The original script called 'call_gemini_thinking', suggesting a specific model or endpoint.
    # We'll use the configured QA/Thinking model name here.
    prompt = f'''
Revise the audio captions based on the input audio and visual captions
Audio Caption:
{audio_captions_formatted}
Visual Caption:
{video_captions_content}
'''
    contents = [prompt] # Simple text prompt for this revision task

    response = utils.call_gemini_api(
        contents=contents,
        api_key=config.GEMINI_API_KEY_REVISION, # Key from original audio_revision.py
        model_name=config.GEMINI_QA_MODEL, # Using QA/Thinking model as per original script's call_gemini_thinking
        system_prompt=config.AUDIO_REVISION_SYSTEM_PROMPT
    )

    if response and response.text:
        revised_caption = response.text.strip()
        print(f"  Generated Revised Audio Caption: {revised_caption[:150]}...")
        if utils.safe_write(output_caption_path, revised_caption + "\n", mode='w'):
            print(f"--- Audio Revision Complete for: {audio_folder_path}. Saved to {os.path.basename(output_caption_path)}. ---")
            return True
        else:
            print(f"  Error: Failed to write revised audio captions to {output_caption_path}.")
            return False
    else:
        print("  Warning: Failed to generate revised audio captions.")
        # Check response details if available
        if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            print(f"  Prompt Feedback: {response.prompt_feedback}")
        if response and hasattr(response, 'candidates') and response.candidates:
             for candidate in response.candidates:
                 if hasattr(candidate, 'finish_reason'):
                     print(f"  Candidate Finish Reason: {candidate.finish_reason.name}")
                 if hasattr(candidate, 'safety_ratings'):
                      print(f"  Candidate Safety Ratings: {candidate.safety_ratings}")
        return False



def perform_av_alignment(folder_path):
    """
    Aligns audio and visual events using the full video and revised captions.
    Reads 'video_consistent_captions.txt' and 'audio_revised_captions.txt'.
    Writes 'av_alignment_captions.txt'.
    """
    print(f"\n--- Starting Audio-Visual Alignment for: {folder_path} ---")
    dir_name = os.path.basename(folder_path)

    # --- Find Full Video ---
    # Use utils.get_video_path for robust finding of the main video file
    video_path = utils.get_video_path(dir_name, base_path=folder_path, segment_index=None, type='video', exact_match=True)
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Full video file not found in {folder_path}. Skipping AV alignment.")
        return False
    print(f"  Using video file: {os.path.basename(video_path)} for alignment.")

    # --- Define Input/Output Paths ---
    visual_caption_path = os.path.join(folder_path, "video_consistent_captions.txt")
    audio_caption_path = os.path.join(folder_path, "audio_revised_captions.txt")
    output_path = os.path.join(folder_path, "av_alignment_captions.txt")

    # --- Check Input Captions ---
    if not os.path.exists(visual_caption_path):
        print(f"Error: Required revised video caption file not found: {visual_caption_path}. Skipping alignment.")
        return False
    if not os.path.exists(audio_caption_path):
        print(f"Error: Required revised audio caption file not found: {audio_caption_path}. Skipping alignment.")
        return False

    # --- Read Captions ---
    video_caption = utils.safe_read(visual_caption_path)
    audio_caption = utils.safe_read(audio_caption_path)
    if video_caption is None or audio_caption is None:
        print("Error: Failed to read revised caption files. Skipping alignment.")
        return False

    # --- Encode Video ---
    # Use base64 encoding as required by the original script's content format
    print("Encoding video for alignment...")
    encoded_video = utils.encode_media(video_path)
    if encoded_video is None:
        print(f"Error encoding video file: {video_path}. Skipping alignment.")
        return False

    # --- Prepare API Call ---
    prompt = f'''
Align the audio and visual events in chronological order.
Video caption:{video_caption}
Audio caption:{audio_caption}
'''
    contents = [
        {
            "mime_type": "video/mp4",
            "data": encoded_video
        },
        prompt
    ]

    print("Input captions loaded. Calling alignment API...")
    # print(f"DEBUG: Alignment Prompt Content:\n{prompt[:500]}...") # Optional debug print

    # --- Call Gemini API ---
    response = utils.call_gemini_api(
        contents=contents,
        api_key=config.GEMINI_API_KEY_ALIGNMENT, # Use dedicated or default key
        model_name=config.GEMINI_ALIGNMENT_MODEL, # Use configured model
        system_prompt=config.AV_ALIGNMENT_SYSTEM_PROMPT
    )

    # --- Process Response ---
    if response and response.text:
        aligned_events = response.text.strip()
        print(f"  Generated Alignment: {aligned_events[:150]}...")
        if utils.safe_write(output_path, aligned_events + "\n", mode='w'):
            print(f"--- Audio-Visual Alignment Complete for: {folder_path}. Saved to {os.path.basename(output_path)}. ---")
            return True
        else:
            print(f"  Error: Failed to write alignment captions to {output_path}.")
            return False
    else:
        print("  Warning: Failed to generate alignment.")
        utils.log_gemini_failure_details(response) # Add detailed logging if possible
        return False