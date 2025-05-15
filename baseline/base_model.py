import dashscope
import os
import base64 
import time
import tqdm
import random
import pathlib 
import json
import multiprocessing
from openai import OpenAI
from v_caption import get_visual_caption 
from a_caption import get_audio_caption, get_speech
from v_event import extract_frames, locate_event, audio_segment, video_segment, get_seg_audio_caption, get_seg_speech, get_seg_visual_caption
import ast
from variables import base_path, json_file_path, fps, max_duration, dashscope_apikey ,max_process
import cv2
dashscope.api_key = dashscope_apikey 

def get_video_duration_seconds(video_path):
    """Helper function to get video duration using OpenCV."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path} with OpenCV.")
            return None
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if video_fps > 0 and frame_count > 0:
            duration = frame_count / video_fps
        else:
            print(f"Warning: Could not get valid FPS ({video_fps}) or frame count ({frame_count}) for {video_path}")
            duration = None
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting video duration for {video_path} using OpenCV: {e}")
        return None

def naive_event_retrieval(video_id, question, choices, max_retries=4, base_delay=2):
    video_path = os.path.join(base_path, video_id, f'{video_id}_video.mp4')
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist for naive_event_retrieval: {video_path}")
        return []

    system_prompt = """
You will be given a video and a question and choices about it. You task is to find out which events are relevant to the question.
For each relevant event, provide its start and end time in seconds along with a brief description of the event.
You should give the events times as a json list:
[{"BEGIN_TIME":begin_time_of_event1, "END_TIME": end_time_of_event1, "EVENT_NAME":brief_description_of_event1}, ...]
"""
    prompt_text = f'Find the relevant events to the question "{question}" with choices "{choices}" and give their time.'
    messages = [
        {"role": "system", "content": [{"type": 'text', "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps},
            {"type": "text", "text": prompt_text}
        ]},
    ]

    for attempt in range(max_retries):
        try:
            response = dashscope.MultiModalConversation.call(model='qwen2.5-vl-7b-instruct', messages=messages)
            result_text = response.output.choices[0].message.content[0]["text"]
            # Clean up common markdown for JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            event_pairs_raw = json.loads(result_text.strip())
            
            processed_event_pairs = []
            video_duration = get_video_duration_seconds(video_path)

            for event_data in event_pairs_raw:
                try:
                    start_time = float(event_data.get('BEGIN_TIME', 0))
                    end_time = float(event_data.get('END_TIME', 0))
                    event_name = event_data.get('EVENT_NAME', "Unknown event")

                    # Basic sanity checks and clamping
                    if video_duration is not None:
                        end_time = min(end_time, video_duration)
                    start_time = max(0, start_time)
                    
                    if start_time > end_time: # If start is after end, maybe make it a zero-duration or skip
                        print(f"Warning: Naive event for {video_id} has start_time ({start_time}) > end_time ({end_time}). Clamping start_time.")
                        start_time = end_time # Or one might choose to skip this event entirely

                    processed_event_pairs.append({
                        'BEGIN_TIME': start_time,
                        'END_TIME': end_time,
                        'EVENT_NAME': event_name
                    })
                except (ValueError, TypeError) as e_parse:
                    print(f"Warning: Could not parse event time for {video_id}: {event_data}. Error: {e_parse}")
            
            print(f"Naive events for {video_id}: {processed_event_pairs}")
            return processed_event_pairs
        except json.JSONDecodeError as e_json:
            print(f"Error decoding JSON from naive_event_retrieval for {video_id}: {e_json}. Raw text: '{result_text if 'result_text' in locals() else 'N/A'}'")
            # Don't retry on JSON decode error, it's likely a model format issue
            return []
        except Exception as e:
            print(f"Error in naive_event_retrieval for {video_id} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
            else:
                print(f"Max retries exceeded for naive_event_retrieval on {video_id}.")
    return []

def load_json_data(file_path):
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

def evaluate_answer(model_answer, correct_answer):
    if not model_answer or not isinstance(model_answer, str):
        return False
    return model_answer.strip().upper().startswith(correct_answer.strip().upper())

def event_retrieval(question, choices, visual_caption, audio_caption, transcription, max_retries=4, base_delay=2):
    system_prompt = f"""
You will be given a question about a video. To answer it, you need visual and audio information of the video.
You will be given the visual and audio description of the video in chronological order. However, you don't know exactly when each visual event and audio event happened at the same time.
Your task is to choose a small set of *visual* events. For each visual event you choose, we will figure out what audio events happened at the same time. The set of visual events you choose should be just enough so that knowing the audio events that happened with them lets you answer the video question.
### Guidelines:
- For each visual event, focus on one movement or change in scene. Never describe a long process.
- Focus on describing the change in frames so that we can retrieve the event precisely.
- Give visual events with small durations so that we can find the audio events that happened exactly at the same time.
- If the visual and audio description are enough for answering the question, give an empty list:[]
### Output Format
Give the events in the format of list:
['event1', 'event2', ...]
If you don't need any event, give an empty list:[] 
Don't generate explanation.
"""
    prompt = f'''
Given the following visual-audio caption, the question and choices of a video, find the visual events required for alignment:
Visual description: {visual_caption}
Audio description: {audio_caption}
Speech in the audio:{transcription}
Question: {question}
Choices: {choices}
'''
    client = OpenAI(api_key=dashscope.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen2.5-14b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=4000
            )
            content = response.choices[0].message.content
            # Attempt to parse the string representation of a list
            try:
                events_list = ast.literal_eval(content.strip())
                if isinstance(events_list, list):
                    return events_list
                else:
                    print(f"Warning: Event retrieval did not return a list for '{question}'. Got: {events_list}")
                    return [] # Return empty if not a list
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse event list from model for '{question}'. Raw content: '{content}'")
                return [] # Return empty if parsing fails
        except Exception as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"General HTTP Error in event_retrieval: {e} Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

    print(f"Max retries exceeded for event_retrieval on question: {question}")
    return [] # Return empty list on failure

def align_events(video_id, event_description, max_retries=4, base_delay=2):
    video_path = os.path.join(base_path, video_id, f'{video_id}_video.mp4')
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist for align_events: {video_path}")
        return None # Return None if video not found
    # Assuming locate_event is robust and handles its own retries or returns None/dict
    return locate_event(event_description, video_path)
 

def ask_model(video_id, question, choices, segment_duration, max_retries=4, base_delay=2):
    # Initialize counts for successful alignments
    num_successful_naive_aligns = 0
    num_successful_smart_aligns = 0

    system_prompt_qa = f"""
You are an expert in understanding videos and answering questions about them.
Your task is to accurately answer multiple-choice questions based on the given video's visual and audio captions.
You will be provided with a question and four choices. You should carefully analyze both the visual and audio captions, and make the most appropriate choice.
Select the single most accurate answer from the given choices. If you are not provided with enough information, please make the most reasonable inference based on the available information.
### Guidelines
- You will be provided with the following information: the visual and audio description of the video, and the transcription of the speech or singing.
- You may be provided with visual-audio event pairs which happpened at the same time. They may be helpful to answer the given question but not always.
- You should carefully analyze the temporal information from the visual and audio description and the aligned visual-audio event pairs to answer questions 
### Output Format
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate explanations.
"""
    client = OpenAI(api_key=dashscope.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # Get base captions
    raw_visual_caption = get_visual_caption(video_id)
    timed_visual_caption = add_time(raw_visual_caption, segment_duration)
    raw_audio_caption = get_audio_caption(video_id)
    timed_audio_caption = add_time(raw_audio_caption, segment_duration)
    raw_transcription = get_speech(video_id)
    timed_transcription = add_time(raw_transcription, segment_duration)
    
    # Get consistent visual caption
    consistent_visual_caption = visual_consistency(video_id, timed_visual_caption) # This one uses timed_visual_caption
    if not consistent_visual_caption: # Fallback if visual_consistency fails
        consistent_visual_caption = timed_visual_caption

    # Naive event retrieval
    naive_events = naive_event_retrieval(video_id, question, choices)
    
    # "Smart" event retrieval
    aligning_events_descriptions = event_retrieval(question, choices, consistent_visual_caption, timed_audio_caption, timed_transcription)
    
    print(f"Video ID: {video_id}")
    print(f"  Naive events raw: {naive_events}")
    print(f"  Aligning events descriptions: {aligning_events_descriptions}")

    audio_path = os.path.join(base_path, video_id, f'{video_id}_audio.wav')
    video_path = os.path.join(base_path, video_id, f'{video_id}_video.mp4')
    
    # Process "Smart" aligned events
    smart_va_events_str = ''
    if aligning_events_descriptions and isinstance(aligning_events_descriptions, list):
        for event_desc in aligning_events_descriptions:
            if not isinstance(event_desc, str): # Skip if not a string description
                continue
            aligned_event_times = align_events(video_id, event_desc) # This calls locate_event
            if aligned_event_times and isinstance(aligned_event_times, dict) and \
               'BEGIN_TIME' in aligned_event_times and 'END_TIME' in aligned_event_times:
                
                start_t, end_t = aligned_event_times['BEGIN_TIME'], aligned_event_times['END_TIME']
                event_duration = end_t - start_t
                
                if 0 < event_duration <= max_duration:
                    num_successful_smart_aligns += 1
                    aud_seg_path = audio_segment(audio_path, start_t, end_t)
                    vid_seg_path = video_segment(video_path, start_t, end_t) # Assuming video_segment returns path
                    
                    if aud_seg_path and vid_seg_path:
                        aud_seg_caption = get_seg_audio_caption(aud_seg_path)
                        # Pass consistent_visual_caption for context if get_seg_visual_caption uses it
                        vid_seg_caption = get_seg_visual_caption(vid_seg_path, consistent_visual_caption) 
                        seg_transcript = get_seg_speech(aud_seg_path)
                        smart_va_events_str += (f'{vid_seg_caption if vid_seg_caption else event_desc} -- {aud_seg_caption} -- "{seg_transcript}";\n')
                        
                        # Cleanup segment files
                        for _ in range(5): # Retry logic for deletion
                            try: 
                                if os.path.exists(aud_seg_path): os.remove(aud_seg_path)
                                if os.path.exists(vid_seg_path): os.remove(vid_seg_path)
                                break
                            except OSError: time.sleep(0.1) 
                else:
                    print(f"  Smart aligned event for '{event_desc}' skipped due to duration: {event_duration:.2f}s (max: {max_duration}s)")
            else:
                print(f"  Failed to align/get times for smart event: {event_desc}")

    # Process Naive aligned events
    naive_va_events_str = ''
    if naive_events and isinstance(naive_events, list):
        for naive_event in naive_events:
            if isinstance(naive_event, dict) and 'BEGIN_TIME' in naive_event and 'END_TIME' in naive_event:
                start_t, end_t = naive_event['BEGIN_TIME'], naive_event['END_TIME']
                event_duration = end_t - start_t
                event_name = naive_event.get('EVENT_NAME', "Event")

                if 0 < event_duration <= max_duration:
                    num_successful_naive_aligns += 1
                    aud_seg_path = audio_segment(audio_path, start_t, end_t)
                    if aud_seg_path: # No video segment needed for naive based on original logic
                        aud_seg_caption = get_seg_audio_caption(aud_seg_path)
                        seg_transcript = get_seg_speech(aud_seg_path)
                        naive_va_events_str += (f"{event_name} -- {aud_seg_caption} -- '{seg_transcript}';\n")
                        for _ in range(5): # Retry logic for deletion
                            try: 
                                if os.path.exists(aud_seg_path): os.remove(aud_seg_path)
                                break
                            except OSError: time.sleep(0.1)
                else:
                     print(f"  Naive event '{event_name}' skipped due to duration: {event_duration:.2f}s (max: {max_duration}s)")
            else:
                print(f"  Malformed naive event data: {naive_event}")

    # Construct prompts for QA model
    # Base Prompt (Original Visual Caption)
    base_prompt_content = f"""
Answer the question about a video below. Give the most reasonable answer.
Visual Description: {timed_visual_caption}
Audio Description: {timed_audio_caption}
Transcription of speech or singing: {timed_transcription}
Question: {question}
Choices: {choices}
"""
    # Consistent Visual Caption Prompt
    consistent_prompt_content = f"""
Answer the question about a video below. Give the most reasonable answer.
Visual Description: {consistent_visual_caption}
Audio Description: {timed_audio_caption}
Transcription of speech or singing: {timed_transcription}
Question: {question}
Choices: {choices}
"""
    # Smart Align Prompt (uses consistent visual caption + smart VA events)
    smart_align_prompt_content = f"""
Answer the question about a video below. Give the most reasonable answer.
Visual Description: {consistent_visual_caption}
Audio Description: {timed_audio_caption}
Transcription of speech or singing: {timed_transcription}
"""
    if smart_va_events_str:
        smart_align_prompt_content += f"""
Each of the following lines contain a visual event and audio event that occur at the same time in the video. They will be given in this format: visual event -- audio_event -- speech_transcription_in_the_audio
{smart_va_events_str}"""
    smart_align_prompt_content += f"""
Question: {question}
Choices: {choices}
"""
    # Naive Align Prompt (uses consistent visual caption + naive VA events)
    naive_align_prompt_content = f"""
Answer the question about a video below. Give the most reasonable answer.
Visual Description: {consistent_visual_caption}
Audio Description: {timed_audio_caption}
Transcription of speech or singing: {timed_transcription}
"""
    if naive_va_events_str:
        naive_align_prompt_content += f"""
Each of the following lines contain a visual event and audio event that occur at the same time in the video. They will be given in this format: visual event -- audio_event -- speech_transcription_in_the_audio
{naive_va_events_str}"""
    naive_align_prompt_content += f"""
Question: {question}
Choices: {choices}
"""
    print(f"  Smart VA events string for QA: {smart_va_events_str.strip() if smart_va_events_str else 'None'}")
    print(f"  Naive VA events string for QA: {naive_va_events_str.strip() if naive_va_events_str else 'None'}")

    results = ["error"] * 4 # Default to error for all
    prompts_to_run = {
        0: ("Smart Align", smart_align_prompt_content),
        1: ("Naive Align", naive_align_prompt_content),
        2: ("Base", base_prompt_content),
        3: ("Consistent", consistent_prompt_content)
    }

    for attempt in range(max_retries):
        all_successful_this_attempt = True
        try:
            for i, (name, content) in prompts_to_run.items():
                # Only rerun if previous attempt for this specific prompt failed (or first attempt)
                if results[i] == "error" or attempt == 0: 
                    # print(f"    Running {name} prompt (Attempt {attempt+1})")
                    response = client.chat.completions.create(
                        model="qwen2.5-14b-instruct",
                        messages=[
                            {"role": "system", "content": system_prompt_qa},
                            {"role": "user", "content": content},
                        ],
                        stream=False, max_tokens=10 # Answer is just a letter
                    )
                    results[i] = response.choices[0].message.content.strip()
            
            if all(r != "error" for r in results): # If all prompts succeeded
                return tuple(results) + (num_successful_smart_aligns, num_successful_naive_aligns)

        except Exception as e:
            all_successful_this_attempt = False
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"General HTTP Error in ask_model QA part: {e}. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries -1:
                time.sleep(delay)
            else:
                break # Break from retry loop if max retries for the batch

    print(f"Max retries exceeded or persistent error in ask_model QA part for question: {question}")
    return tuple(results) + (num_successful_smart_aligns, num_successful_naive_aligns)


def visual_consistency(video_id, video_caption, max_retries=5, base_delay=4):
    video_path = os.path.join(base_path, video_id, f'{video_id}_video.mp4')
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist for visual_consistency: {video_path}")
        return video_caption # Return original caption if video missing

    system_prompt = f"""
### Task:
You are an expert in understanding scene transitions based on visual features in a video. You are required to improve the consistency of the descriptions of the video segments. You will get the video and the visual captions of its segments. You should check whether the objects mentioned in the captions are the same object. For example, if the first segment mentions a person while the second segment also mentions a person, you should check the video to find out whether they are the same person. You should revise the captions to show their relations.
#### Guidelines For Improving Consistency:
- Find the persons mentioned in the captions of different segment. Use the video to adjust the captions to explicitly note relationships, continuity, or differences between persons across segments.
- If the captions mentioned the environments of the segments, determine whether they are the same environment. Adjust the captions accordingly.
- Find the import objects mentioned in the captions of different segment. Use the video to adjust the captions to explicitly note relationships, continuity, or differences between objects across segments.
- Don't revise other parts of the captions, make sure the captions are detailed.
#### Output format
Your output should be in the same form as the input(plain text):

0-10s: In this section, ...
10-20s: In this section, ...
...
"""
    client = OpenAI(api_key=dashscope.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    prompt_text = f'Video segment description:{video_caption}'
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen2.5-vl-7b-instruct", # This is a multimodal model
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": [
                              {"type": "video", "video": video_path, 'fps': fps},
                              {"type": "text", "text": prompt_text},
                          ]}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error while getting visual consistency for {video_id} (attempt {attempt + 1}): {e}")
            if attempt < max_retries -1:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0,1))
            else:
                print(f"Max retries exceeded for visual_consistency on {video_id}")
                return video_caption # Return original caption on failure

def process_item(item_tuple): # Expects (item, failed_counter) for multiprocessing
    item, failed_counter = item_tuple
    try:
        question = item.get('Question')
        choices = item.get('Choice') # Assuming it's a string like "A) ..., B) ..."
        correct_answer_letter = item.get('Answer') # Assuming this is 'A', 'B', 'C', or 'D'
        video_id = item.get('video_id')
        
        video_duration_str = item.get('video_duration')
        if video_duration_str == '30s':
            segment_duration = 10
        elif video_duration_str == '60s':
            segment_duration = 20
        else:
            print(f"Warning: Unknown video_duration '{video_duration_str}' for {video_id}. Defaulting segment_duration to 15.")
            segment_duration = 15 # A reasonable default

        if not all([question, choices, correct_answer_letter, video_id]):
            print(f"Warning: Skipping item due to missing required fields. Item: {item.get('video_id', 'Unknown ID')}")
            return False, False, False, False, 0, 0 # Answers, smart_align_count, naive_align_count

        (smart_align_answer, naive_align_answer, 
         base_answer, consistent_answer,
         num_successful_smart, num_successful_naive) = ask_model(video_id, question, choices, segment_duration)
        
        smart_align_is_correct = evaluate_answer(smart_align_answer, correct_answer_letter)
        naive_align_is_correct = evaluate_answer(naive_align_answer, correct_answer_letter)
        base_is_correct = evaluate_answer(base_answer, correct_answer_letter)
        consistent_is_correct = evaluate_answer(consistent_answer, correct_answer_letter)

        print(f"\nProcessed Video ID: {video_id}, Question: {question[:50]}...")
        print(f"  Choices: {choices}")
        print(f"  Correct Answer: {correct_answer_letter}")
        print(f"  Base Model Answer: '{base_answer}' ({'Correct' if base_is_correct else 'Incorrect'})")
        print(f"  Consistent Model Answer: '{consistent_answer}' ({'Correct' if consistent_is_correct else 'Incorrect'})")
        print(f"  Naive Align Model Answer: '{naive_align_answer}' ({'Correct' if naive_align_is_correct else 'Incorrect'}) - Successful Naive Aligns: {num_successful_naive}")
        print(f"  Smart Align Model Answer: '{smart_align_answer}' ({'Correct' if smart_align_is_correct else 'Incorrect'}) - Successful Smart Aligns: {num_successful_smart}")
        
        return (smart_align_is_correct, naive_align_is_correct, base_is_correct, consistent_is_correct,
                num_successful_smart, num_successful_naive)

    except Exception as e:
        print(f"!!! Unhandled Error processing item: {item.get('video_id', 'Unknown ID')}. Error: {e}")
        failed_counter.value += 1
        return False, False, False, False, 0, 0

def add_time(caption, segment_duration):
    if not caption or not isinstance(caption, str):
        return "" # Return empty if caption is None or not a string
    lines = caption.strip().splitlines()
    timed_lines = []
    for i, line in enumerate(lines):
        if line.strip(): # Add time only to non-empty lines
            timed_lines.append(f"{i*segment_duration}-{(i+1)*segment_duration}s: {line.strip()}")
    return '\n'.join(timed_lines)


def test_all_questions(file_path):
    data = load_json_data(file_path)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # data = data[:20] # Limiting for testing, remove for full run
    total_questions_to_process = len(data)
    print(f"Starting to process {total_questions_to_process} questions...")

    manager = multiprocessing.Manager()
    failed_counter = manager.Value('i', 0)
    
    # Prepare items for pool.map, including the shared counter
    items_with_counter = [(item, failed_counter) for item in data]

    num_processes = min(os.cpu_count() or 1, max_process) 
    print(f"Using {num_processes} processes for parallel execution.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # results will be a list of tuples: (bool, bool, bool, bool, int, int)
        results = list(tqdm.tqdm(pool.imap(process_item, items_with_counter), total=total_questions_to_process))

    # Initialize result aggregation structures
    VIDEO_CAT = sorted(list(set(item.get('video_category', 'Unknown') for item in data)))
    QA_TYPE = sorted(list(set(item.get('Type', 'Unknown') for item in data)))
    VIDEO_DURATION_CATS = sorted(list(set(item.get('video_duration', 'Unknown') for item in data)))
    
    all_categories = VIDEO_CAT + QA_TYPE + VIDEO_DURATION_CATS
    
    metrics = {
        "smart_align_correct": {cat: 0 for cat in all_categories},
        "naive_align_correct": {cat: 0 for cat in all_categories},
        "base_correct": {cat: 0 for cat in all_categories},
        "consistent_correct": {cat: 0 for cat in all_categories},
        "num_successful_smart_aligns": {cat: 0 for cat in all_categories},
        "num_successful_naive_aligns": {cat: 0 for cat in all_categories},
        "qa_number": {cat: 0 for cat in all_categories}
    }

    # Aggregate results
    for i, item_data in enumerate(data):
        if i < len(results): # Ensure we don't go out of bounds if some items failed catastrophically early
            res_tuple = results[i]
            if len(res_tuple) == 6: # Check for correct tuple length
                sa_correct, na_correct, b_correct, c_correct, num_smart, num_naive = res_tuple
                
                current_cats = [
                    item_data.get('video_category', 'Unknown'),
                    item_data.get('Type', 'Unknown'),
                    item_data.get('video_duration', 'Unknown')
                ]
                
                for cat_key in current_cats:
                    if cat_key not in metrics["qa_number"]: continue # Should not happen if initialized correctly

                    metrics["qa_number"][cat_key] += 1
                    if sa_correct: metrics["smart_align_correct"][cat_key] += 1
                    if na_correct: metrics["naive_align_correct"][cat_key] += 1
                    if b_correct: metrics["base_correct"][cat_key] += 1
                    if c_correct: metrics["consistent_correct"][cat_key] += 1
                    metrics["num_successful_smart_aligns"][cat_key] += num_smart
                    metrics["num_successful_naive_aligns"][cat_key] += num_naive
            else:
                print(f"Warning: Result tuple for item {i} has unexpected length: {len(res_tuple)}")
        else:
            print(f"Warning: Missing result for item {i}")


    print("\n--- Results Summary ---")
    category_sets = [
        ("Video Category", VIDEO_CAT),
        ("QA Type", QA_TYPE),
        ("Video Duration", VIDEO_DURATION_CATS)
    ]

    for setName, cat_list in category_sets:
        print(f"\n--- {setName} ---")
        for category_key in cat_list:
            q_num = metrics["qa_number"].get(category_key, 0)
            if q_num == 0:
                print(f"  {category_key}: No questions processed.")
                continue
            
            print(f"  {category_key} (Total: {q_num}):")
            print(f"    Base Model Accuracy:         {metrics['base_correct'][category_key]}/{q_num} ({metrics['base_correct'][category_key]/q_num*100:.2f}%)")
            print(f"    Consistent Model Accuracy:   {metrics['consistent_correct'][category_key]}/{q_num} ({metrics['consistent_correct'][category_key]/q_num*100:.2f}%)")
            print(f"    Naive Align Model Accuracy:  {metrics['naive_align_correct'][category_key]}/{q_num} ({metrics['naive_align_correct'][category_key]/q_num*100:.2f}%)")
            print(f"      - Avg Successful Naive Aligns: {metrics['num_successful_naive_aligns'][category_key]/q_num:.2f}")
            print(f"    Smart Align Model Accuracy:  {metrics['smart_align_correct'][category_key]}/{q_num} ({metrics['smart_align_correct'][category_key]/q_num*100:.2f}%)")
            print(f"      - Avg Successful Smart Aligns: {metrics['num_successful_smart_aligns'][category_key]/q_num:.2f}")

    # Overall Totals
    total_processed_effectively = sum(1 for res_tuple in results if len(res_tuple) == 6) # Count only fully processed items
    if total_processed_effectively == 0:
        print("\nNo questions were effectively processed. Cannot calculate overall accuracy.")
    else:
        overall_base_correct = sum(metrics["base_correct"].values())
        overall_consistent_correct = sum(metrics["consistent_correct"].values())
        overall_naive_align_correct = sum(metrics["naive_align_correct"].values())
        overall_smart_align_correct = sum(metrics["smart_align_correct"].values())
        
        # The qa_number for overall needs to be summed carefully if categories overlap,
        # but since we iterate through `data` once, total_processed_effectively is the right denominator.
        # For per-category sums, they are independent.
        # The sum of all q_num in metrics["qa_number"] will be total_questions_to_process * number_of_categorizations (e.g., 3)
        # So, we use total_questions_to_process (or total_processed_effectively) for overall.

        # Total correct answers for overall accuracy calculation:
        total_base_correct_overall = sum(r[2] for r in results if len(r)==6)
        total_consistent_correct_overall = sum(r[3] for r in results if len(r)==6)
        total_naive_align_correct_overall = sum(r[1] for r in results if len(r)==6)
        total_smart_align_correct_overall = sum(r[0] for r in results if len(r)==6)
        
        total_successful_naive_aligns_overall = sum(r[5] for r in results if len(r)==6)
        total_successful_smart_aligns_overall = sum(r[4] for r in results if len(r)==6)


        print("\n--- Overall Performance ---")
        print(f"Total Questions Processed: {total_questions_to_process}")
        print(f"Total Questions Effectively Processed (returned full results): {total_processed_effectively}")
        if total_processed_effectively > 0:
            print(f"  Base Model Accuracy:         {total_base_correct_overall}/{total_processed_effectively} ({total_base_correct_overall/total_processed_effectively*100:.2f}%)")
            print(f"  Consistent Model Accuracy:   {total_consistent_correct_overall}/{total_processed_effectively} ({total_consistent_correct_overall/total_processed_effectively*100:.2f}%)")
            print(f"  Naive Align Model Accuracy:  {total_naive_align_correct_overall}/{total_processed_effectively} ({total_naive_align_correct_overall/total_processed_effectively*100:.2f}%)")
            print(f"    - Total Successful Naive Aligns: {total_successful_naive_aligns_overall} (Avg: {total_successful_naive_aligns_overall/total_processed_effectively:.2f})")
            print(f"  Smart Align Model Accuracy:  {total_smart_align_correct_overall}/{total_processed_effectively} ({total_smart_align_correct_overall/total_processed_effectively*100:.2f}%)")
            print(f"    - Total Successful Smart Aligns: {total_successful_smart_aligns_overall} (Avg: {total_successful_smart_aligns_overall/total_processed_effectively:.2f})")
    
    print(f"\nTotal items that failed catastrophically during processing: {failed_counter.value}")


if __name__ == "__main__":
    # Ensure your 'variables.py' defines:
    # base_path (str): Path to the directory containing video_id subfolders.
    # json_file_path (str): Path to your input JSON file.
    # fps (int): Frames per second to use for video processing (e.g., for qwen-vl).
    # max_duration (int/float): Maximum duration in seconds for a segment to be considered for alignment.
    
    # Example check for variables.py content (you would have these defined)
    # if not all([isinstance(base_path, str), isinstance(json_file_path, str), 
    #             isinstance(fps, int), isinstance(max_duration, (int, float))]):
    #     print("Error: One or more required variables (base_path, json_file_path, fps, max_duration) not correctly defined.")
    #     exit(1)

    test_all_questions(json_file_path)