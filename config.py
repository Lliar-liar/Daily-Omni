# config.py

import os

# --- API Keys ---
# WARNING: Storing API keys directly in code is not recommended for production.
# Consider using environment variables or a secrets management system.
GEMINI_API_KEY_DEFAULT = 'YOUR_API_KEY' # Default key, can be overridden
GEMINI_API_KEY_AUDIO = 'YOUR_API_KEY' # Key used in original audio_caption.py
GEMINI_API_KEY_REVISION = 'YOUR_API_KEY' # Key used in original audio_revision.py
GEMINI_API_KEY_ALIGNMENT = 'YOUR_API_KEY'
GEMINI_API_KEY_QA = 'YOUR_API_KEY' # Key used in original advance_QA_gen.py (Thinking)



# Keys for alternative QA generation API (DeepSeek/VolcEngine)
# Using placeholder names - adjust if needed
DEEPSEEK_API_KEY = 'YOUR_API_KEY' 
DEEPSEEK_BASE_URL = "YOUR_BASE_URL" 
FILTER_MODEL_1_API_KEY='YOUR_API_KEY'
FILTER_MODEL_1_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" # As an example
FILTER_MODEL_2_API_KEY='YOUR_API_KEY'
FILTER_MODEL_2_BASE_URL="YOUR_BASE_URL" 
# --- Model Names ---
GEMINI_VISION_MODEL = 'gemini-2.0-flash' # Used for video captioning
GEMINI_AUDIO_MODEL = 'gemini-2.0-flash' # Used for audio captioning 
GEMINI_ALIGNMENT_MODEL = 'gemini-2.0-flash' # Model for AV Alignment 
GEMINI_REVISION_MODEL = 'gemini-2.0-flash' 
GEMINI_QA_MODEL = 'gemini-2.0-flash' 


DEEPSEEK_QA_MODEL = "deepseek-r1" #model name or instance id
FILTER_MODEL_1_NAME="deepseek-v3"
FILTER_MODEL_2_NAME="gpt-4o-ca"
# --- Global Settings ---

SEGMENT_DURATION = 10 
BASE_DIR = "./example_videos" 
CSV_PATH = "./example_metadata.csv" 
MAX_WORKERS_THREADS=2
MAX_WORKERS_PROCESSES=1 #Set to None for no limit
# --- Output File Paths ---
CONSOLIDATED_QA_FILENAME = "./qa_example.json"  # In the current run directory
FILTERED_QA_FILENAME = "./qa_example_filtered.json" # In the current run directory
# --- Retry Configuration ---
MAX_RETRIES = 5
BASE_DELAY = 1 # seconds

# --- System Prompts (Keep original text exactly) ---

VIDEO_CAPTION_SYSTEM_PROMPT = '''
### Task:
You are an expert in understanding scene transitions based on visual features in a video. You are requested to create the descriptions for the current clip sent to you, which includes multiple sequential frames.
#### Guidelines For Clip Description:
- Analyze the narrative progression implied by the sequence of frames, interpreting the sequence as a whole.
- If text appears in the frames, you must describe the text in its original language and provide an English translation in parentheses. For example: 书本 (book). Additionally, explain the meaning of the text within its context.
- When referring to people, use their characteristics, such as clothing, to distinguish different people.
- **IMPORTANT** Please provide as many details as possible in your description, including colors, shapes, and textures of objects, actions and characteristics of humans, as well as scenes and backgrounds.
#### Output format
Your output should look like this:"In this section, ... . Then, .... Finally, ...."
'''

AUDIO_CAPTION_SYSTEM_PROMPT = '''
You are an expert in understanding audio information in a video. You are requested to create the descriptions for the audio clip sent to you.
### Guidelines For Clip Description:
- Describe the sounds(human speech, music and other sound) in the audio clip and their type in time order.
- When there's speech in the clip, describe who is speaking, the speaker's tone and what he/she/they are saying.
- Explicitly keep the transcription of the speech in you caption.
- If the speech is not in English, translate it to English.
- When there's music in the clip, describe the music's genre, it's tone and what instrument is used to play it.
- When there's other sounds in the clip, describe the sounds and their type.
### Output format
Your output should look like this:
In this section, ... . Then, .... Finally, ....
'''

VIDEO_REVISION_SYSTEM_PROMPT = f'''
### Task:
You are an expert in understanding scene transitions based on visual features in a video. You are required to improve the consistency of the descriptions of the video segments. You will get the video and the visual captions of its segments. You should check whether the objects mentioned in the captions are the same obejct. For example, if the first segment mentions a person while the second segment also mentions a person, you should check the video to find out whether they are the same person. You should revise the captions to show their relations.
#### Guidelines For Improving Consistency:
- Find the persons mentioned in the captions of different segment. Use the video to adjust the captions to explicitly note relationships, continuity, or differences between persons across segments.
- If the captions mentioned the environments of the segments, determine whether they are the same environment. Adjust the captions accordingly.
- Find the import objects mentioned in the captions of different segment. Use the video to adjust the captions to explicitly note relationships, continuity, or differences between objects across segments.
- Don't revise other parts of the captions, make sure the captions are detailed.
#### Output format
Your output should be in the same form as the input(plain text):

0-{SEGMENT_DURATION}s: In this section, ...
{SEGMENT_DURATION}-{SEGMENT_DURATION*2}s: In this section, ...
...

'''

# Note: audio_revision.py had two system prompts. Combining the core revision task prompt here.
# The audio_consistency prompt seemed like a sub-task or alternative. Keeping the main one.
AUDIO_REVISION_SYSTEM_PROMPT = f"""
You are an expert in revising the audio captions of a video. You will get the visual captions and audio captions of the segments of a video. However, the audio captions might not be accurate. They may contain errors such as misidentifying sounds or attributing incorrect sources to sounds. You should revise the audio captions to correct the errors.
### Guidelines For Revision:
- Analyze the audio and visual captions carefully, check if the audio captions contain misidentifying of sounds.
- Determine misidentification only when the sound described in the audio caption closely resembles a sound that can be reasonably inferred from the visual caption. For example, if the visual caption describes a skateboard dropping to the ground, and the audio caption interprets it as a door slamming, this would be considered a misidentification due to the similarity in sound.
- Revise the audio captions to accurately reflect the correct sound and its possible source.
### Output Format
Only output the revised audio caption. Your output should be in the following format(plain text):

0-{SEGMENT_DURATION}s: In this section,...
{SEGMENT_DURATION}-{SEGMENT_DURATION*2}s: In this section,...
...

"""

ADVANCED_QA_SYSTEM_PROMPT = '''
### Task:
Given a detailed description that summarizes the visual and audio(sound event sequence and speech) content of a video and a series of audio and visual events that occurs at the same time, generate question-answer pairs that based on the description to help human better understand the video.
#### Guidelines For Question-Answer Pairs Generation:
- The QAs you generate should be answered with and only with BOTH audio(human speech and object sound) and visual information. It shouldn't be answered correctly only with audio or video information alone.
- When generating choices, make sure the choices are equally long so that there won't be data bias.
- To increase the difficulty, generate answer choices with some ambiguity or confusion, ensuring that it requires careful attention to both visual and audio elements to answer correctly.
- Make the answer choices more deceptive so that multiple options appear plausible, rather than having only one or two clearly reasonable choices. This will increase the question's difficulty.
- For all types of questions, don't explictly mention too much information in the question otherwise the question can be answered correctly without using information from the video.
### Question Types:
You should generate the following types of questions:
- AV Event Alignment: Formulate questions to determine which audio and visual events occurred simultaneously with each other.
- Event Sequence: Formulate questions to determine the temporal sequence of visual and audio events in the video.
- Reasoning: Formulate questions to explain the cause or reason behind the occurrence of a visual or audio event in the video.
- Inference: Formulate questions to speculate on information not explicitly presented in the video.
- Comparative: Formulate questions to compare the similarity or difference between the audio and visual information of two or more events in the video.
- Context understanding: Formulate questions to determine the contextual information surrounding a specific event in the video.
### Output Format:
The questions should be in the form of multiple choice. The choices should look like:["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"]
The answer should be a single capital letter A, B, C, or D.
Your output should be formed in a JSON file.
You response should look like:
```json
[{"Type": <type-1>, "Question": <question-1>, "Choice": <choice-1>, "Answer": <answer-1>, "Explaination": <explanation-1>},
{"Type": <type-2>, "Question": <question-2>, "Choice": <choice-2>, "Answer": <answer-2>, "Explaination": <explanation-2>},
...]
'''
QA_FILTER_SYSTEM_PROMPT = """
You are an expert in understanding videos and answering questions about them. 
You will be provided with a question about a video and a list of choices. However, you won't be provided with the video. So you should use information from the question and choices to guess the answer.
Select the single most plausible answer from the given choices.
### Output Format
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate explainations.
"""
AV_ALIGNMENT_SYSTEM_PROMPT = '''
### Task:
You are an expert in aligning visual and audio events. You will get a video and its visual and audio captions in chronological order. For each audio event in the audio caption, you should determine which visual event occurs simultaneously based on the video.
### Guidelines For Alignment
- Explictly describe the audio and visual events, don't mention excessive details such as the environment.
- For human speech, describe it as: "<brief_transciption_of_speech>" spoken. You can omit part of the speech if it is long.
- For singing, describe it as: "<brief_transciption_of_singing>" sung.
- For object sound, describe it as: Sound of ... occurs.
- The aligned audio and visual events should be output in chronological order.
### Output format
Your output should be in the following form:

<audio_event1> -- <visual_event1>,
<audio_event2> -- <visual_event2>,
...

'''