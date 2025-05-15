# captioning.py

import os
import pathlib
import dashscope
import utils
import config

def generate_video_captions(video_folder_path):
    """
    Generates detailed video captions for segmented video files in a folder using Gemini.
    Saves captions to 'video_detail_captions.txt'.
    """
    print(f"\n--- Starting Video Captioning for: {video_folder_path} ---")
    dir_name = os.path.basename(video_folder_path)
    output_txt_path = os.path.join(video_folder_path, "video_detail_captions.txt")

    # Find video segment files
    video_file_paths = []
    i = 0
    while True:
        potential_video_path = os.path.join(video_folder_path, f"{dir_name}_video_{i}.mp4")
        if os.path.exists(potential_video_path):
            video_file_paths.append(potential_video_path)
            i += 1
        else:
            break # Stop when the next sequential file doesn't exist

    if not video_file_paths:
        print(f"Warning: No video segment files (e.g., '{dir_name}_video_0.mp4') found in {video_folder_path}. Skipping video captioning.")
        return False # Indicate no files processed

    # Clear or create the output file
    if not utils.safe_write(output_txt_path, "", mode='w'):
        print(f"Error: Could not initialize output file {output_txt_path}. Aborting video captioning.")
        return False

    print(f"Found {len(video_file_paths)} video segments. Generating captions...")
    success_count = 0
    for video_path in video_file_paths:
        print(f"Processing segment: {os.path.basename(video_path)}")

        video_bytes = utils.read_media_bytes(video_path)
        if video_bytes is None:
             print(f"Skipping segment due to reading error: {os.path.basename(video_path)}")
             continue
        contents = [
            {"mime_type": "video/mp4", "data": video_bytes},
             "Describe this video clip in detail according to the guidelines." # User prompt
        ]


        response = utils.call_gemini_api(
            contents=contents,
            api_key=config.GEMINI_API_KEY_DEFAULT, # Or specific key if needed
            model_name=config.GEMINI_VISION_MODEL,
            system_prompt=config.VIDEO_CAPTION_SYSTEM_PROMPT
        )

        if response and response.text:
            caption = response.text.replace('\n', ' ').strip() # Clean up caption
            print(f"  Generated Caption: {caption[:100]}...") # Print preview
            if utils.safe_write(output_txt_path, caption + "\n", mode='a'):
                 success_count += 1
            else:
                 print(f"  Error: Failed to write caption for {os.path.basename(video_path)} to file.")
        else:
            print(f"  Warning: Failed to generate caption for {os.path.basename(video_path)}.")
            # Optionally write a placeholder or error message to the file
            # utils.safe_write(output_txt_path, f"Error generating caption for {os.path.basename(video_path)}\n", mode='a')

    print(f"--- Video Captioning Complete for: {video_folder_path}. Generated {success_count}/{len(video_file_paths)} captions. ---")
    return success_count == len(video_file_paths)


def generate_audio_captions(audio_folder_path, use_gemini=True):
    """
    Generates detailed audio captions for segmented audio files in a folder.
    Uses Gemini by default, can optionally use DashScope.
    Saves captions to 'audio_detail_captions.txt'.
    """
    print(f"\n--- Starting Audio Captioning for: {audio_folder_path} (Backend: {'Gemini' if use_gemini else 'DashScope'}) ---")
    dir_name = os.path.basename(audio_folder_path)
    output_txt_path = os.path.join(audio_folder_path, "audio_detail_captions.txt")

    # Find audio segment files
    audio_file_paths = []
    i = 0
    while True:
        # Assuming .wav format based on original script
        potential_audio_path = os.path.join(audio_folder_path, f"{dir_name}_audio_{i}.wav")
        if os.path.exists(potential_audio_path):
            audio_file_paths.append(potential_audio_path)
            i += 1
        else:
            break

    if not audio_file_paths:
        print(f"Warning: No audio segment files (e.g., '{dir_name}_audio_0.wav') found in {audio_folder_path}. Skipping audio captioning.")
        return False

    # Clear or create the output file
    if not utils.safe_write(output_txt_path, "", mode='w'):
        print(f"Error: Could not initialize output file {output_txt_path}. Aborting audio captioning.")
        return False

    print(f"Found {len(audio_file_paths)} audio segments. Generating captions...")
    success_count = 0
    for audio_path in audio_file_paths:
        print(f"Processing segment: {os.path.basename(audio_path)}")
        text_content = None

        if use_gemini:
            # Use raw bytes for Gemini audio input
            audio_bytes = utils.read_media_bytes(audio_path)
            if audio_bytes is None:
                print(f"Skipping segment due to reading error: {os.path.basename(audio_path)}")
                continue

            contents = [
                {"mime_type": "audio/wav", "data": audio_bytes}, # Adjust mime_type if needed
                "Describe the sounds(human speech, music and other sound) in the given audio clip and their type in time order." # User prompt
            ]

            response = utils.call_gemini_api(
                contents=contents,
                api_key=config.GEMINI_API_KEY_AUDIO, # Specific key from original script
                model_name=config.GEMINI_AUDIO_MODEL,
                system_prompt=config.AUDIO_CAPTION_SYSTEM_PROMPT
            )

            if response and response.text:
                text_content = response.text.replace('\n', ' ').strip()
            else:
                print(f"  Warning: Gemini failed to generate caption for {os.path.basename(audio_path)}.")

        else: # Use DashScope
            dashscope.api_key = config.DASHSCOPE_API_KEY
            # DashScope example used file path directly in the 'audio' field
            messages = [
                {"role": "system", "content": [{"text": config.AUDIO_CAPTION_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"audio": audio_path}, # Use file path
                        {"text": "Describe the sounds in the audio clip in time order."}
                    ],
                }
            ]
            try:
                # Adding retry logic might be needed for DashScope too
                response = dashscope.MultiModalConversation.call(
                    model=config.DASHSCOPE_AUDIO_MODEL,
                    messages=messages
                    )
                # Check response status (example, adjust based on actual API)
                if response.status_code == 200 and response.output and response.output.choices:
                     text_content = response.output.choices[0].message.content[0]["text"].replace('\n', ' ').strip()
                else:
                    print(f"  Warning: DashScope call failed for {os.path.basename(audio_path)}. Status: {response.status_code}, Output: {response.output}")
            except Exception as e:
                 print(f"  Error calling DashScope API for {os.path.basename(audio_path)}: {e}")


        # Write content if generated
        if text_content:
            print(f"  Generated Caption: {text_content[:100]}...")
            if utils.safe_write(output_txt_path, text_content + "\n", mode='a'):
                success_count += 1
            else:
                print(f"  Error: Failed to write caption for {os.path.basename(audio_path)} to file.")
        else:
             # Optionally write placeholder if generation failed
             # utils.safe_write(output_txt_path, f"Error generating caption for {os.path.basename(audio_path)}\n", mode='a')
             pass


    print(f"--- Audio Captioning Complete for: {audio_folder_path}. Generated {success_count}/{len(audio_file_paths)} captions. ---")
    return success_count == len(audio_file_paths)