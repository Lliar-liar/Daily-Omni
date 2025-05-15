import json
import os
import cv2
import time
import random
import base64
import tqdm
import multiprocessing
from openai import OpenAI
import openai
from variables import base_path
import re
from pydub import AudioSegment
import dashscope
import requests
from a_caption import encode_audio
from variables import fps,dashscope_apikey,openai_apikey,openai_baseurl
from moviepy.editor import VideoFileClip
dashscope.api_key = dashscope_apikey
def get_video_duration(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"Error in getting video duration: {e}")
        return None
def get_seg_audio_caption(audio_path,max_retries=4,base_delay=2,use_model="qwen2_audio"):
    results=''
    system_prompt='''
Describe the most prominent sounds in the given audio clip in a concise way in a simple sentence. Don't mention the timestamp of the sound.
Your output should be in the format of: 'Sound of ... occur.'
'''
    if use_model=="qwen2_audio":
        audio_base64=encode_audio(audio_path)
        messages = [
            {
                "role": "system",
                # "content": system_prompt,
                "content":"You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type":"audio", "audio_url": f"data:audio/wav;base64,{audio_base64}"},
                    {"type":"text", "text": system_prompt}
                ],
            }
        ]
        # print(messages)
        payload = {
            "conversation": messages,
            "temperature": 0.9
        }
        for attempt in range(max_retries):
            try:
                response = requests.get("http://127.0.0.1:6006/v1/audio/inference",json=payload)
                # print(response.json())
                break
            except Exception as e:
                print(f"Error while getting audio caption: {e}")
                time.sleep(base_delay * (2 ** attempt))
                continue
        # print(response)
        text_content = response.json()['response']
        return text_content
    else:
        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"audio": audio_path},
                    {"text": "Describe the audio clip concisely with the given format."}
                ],
            }
        ]
        for attempt in range(max_retries):
            try:
                response = dashscope.MultiModalConversation.call(model="qwen-audio-turbo-1204", messages=messages)
                return response.output.choices[0].message.content[0]["text"]
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(base_delay * (2 ** attempt))
                continue
        text_content = response.output.choices[0].message.content[0]["text"]
        return text_content

def get_seg_visual_caption(video_path,video_description,max_retries=4,base_delay=2):
    result=''
    system_prompt = f"""
You will be given a segment of a video and the textual description of the complete video clip.
You should describe the video segment concisely.
### Guidelines
- Describe the video segment concisely in one simple sentance
- Stick to the original video description so that we can know which event in the video does the segment contain
- Condense your description and omit unnecessary details
"""
# - Prioritize using verbatim text from the original video description
 
    prompt = f'''
Locate the given video segment in the following video description:
{video_description}

Give the description of the video segment.
'''
    messages= [
        {
            "role": "system",
            "content": [{"type":'text', "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type":"video", "video": video_path,"fps":fps},
                {"type":"text", "text": prompt}
            ],
        }
    ]
    for attempt in range(max_retries):
        try:
            response = dashscope.MultiModalConversation.call(model='qwen2.5-vl-7b-instruct', messages=messages)
            return response.output.choices[0].message.content[0]["text"]
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(base_delay * (attempt + 1))
            

def get_seg_speech(audio_path,max_retries=4,base_delay=2):
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=openai_apikey, base_url=openai_baseurl)
            audio_file= open(audio_path, "rb")
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            return transcription.text
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(base_delay * (2 ** attempt))
            continue

def extract_frames(video_path, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Fail to read: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(video_fps / fps)))  
    base64_images = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_images.append(base64.b64encode(buffer).decode('utf-8'))

        frame_count += 1

    cap.release()
    return base64_images

def audio_segment(audio_path, begin_time, end_time):
    try:
        audio = AudioSegment.from_wav(audio_path)
        start_ms = int(begin_time * 1000)
        end_ms = int(end_time * 1000)

        if start_ms < 0:
            start_ms = 0  
        if end_ms > len(audio):
            end_ms = len(audio) 
        if start_ms >= end_ms:
            print("Begin time should be less than end time.")
            return None

        segment = audio[start_ms:end_ms]

        base, ext = os.path.splitext(audio_path)
        output_path = f"{base}_{begin_time}_{end_time}_segment.wav"

        segment.export(output_path, format="wav")
        return output_path
    except FileNotFoundError:
        print(f"Error: File not found: {audio_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def video_segment(video_path, begin_time, end_time):
    try:
        video = VideoFileClip(video_path)
        start_sec = begin_time
        end_sec = end_time
        segment = video.subclip(start_sec, end_sec)
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_{begin_time}_{end_time}_segment.mp4"
        segment.write_videofile(output_path, codec="mpeg4", audio=False)
        return output_path
    except Exception as e:
        print(f"Error: {e}")
        return None



def get_video_path(video_id, base_path):
    video_path=[]
    for i in range(0,3):
        if os.path.exists(os.path.join(base_path, video_id, f'{video_id}_video_{i}.mp4')):
            video_path.append(os.path.join(base_path, video_id, f'{video_id}_video_{i}.mp4'))
        else: 
            break
    # print(video_path)
    return video_path

def get_audio_path(video_id, base_path=base_path):
    video_path=[]
    for i in range(0,3):
        if os.path.exists(os.path.join(base_path, video_id, f'{video_id}_audio_{i}.wav')):
            video_path.append(os.path.join(base_path, video_id, f'{video_id}_audio_{i}.wav'))
        else: 
            break

    return video_path
def visual_events(video_id, max_retries=4, base_delay=4):
    results=''
    video_path = get_video_path(video_id)
    audio_path = get_audio_path(video_id)
    
    for i, segment in enumerate(video_path):
        time_result=[]
        system_prompt = f"""
### Task:
You are an expert in understanding scene transitions based on visual features in a video. You are required to create a list of events that occur in the video.
### Guidelines For Your Task
- Analyze the narrative progression implied by the sequence of frames, interpreting the sequence as a whole.
- Summarize the events in the video explicitly, and list them in time order. A scene with special features such as a close up of an object should also be counted as an event.
- Describe the events with their prominent features so that other people can know which event is being described.
### Output format
Split the events with ';'. Your description of the video should look like this:
"Event list: EVENT1, EVENT2, EVENT3, ..."
        """

        prompt = 'List the events in the video.'
        # base64_images = extract_frames(segment, fps)
        
        completion = client.chat.completions.create(
            model="qwen2.5-vl-7b-instruct",
            messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "video", "video": segment,'fps':2},
                        {"type": "text", "text": prompt},
                    ]}]
        )
        # print(completion.choices[0].message.content)
        events = re.split(r";\s*", completion.choices[0].message.content.split("Event list: ")[1])
        begin_time=0
        for event in events:
            times=locate_event(event,segment)
            if(times['BEGIN_TIME']>=begin_time and times['END_TIME']-times['BEGIN_TIME']<=4.5):
                time_result.append(times)
                begin_time=times['BEGIN_TIME']
        for times in time_result:
            seg_path=audio_segment(audio_path[i],times['BEGIN_TIME'],times['END_TIME'])
            
            if(seg_path!=None):
                seg_caption=get_audio_caption(seg_path)
                seg_transcript=get_speech(seg_path)
                results+=(f"{times['event']} -- {seg_caption} {seg_transcript};\n")
                for i in range(0,5):
                    try:
                        os.remove(seg_path)
                        break
                    except:
                        time.sleep(2+random.random())
                        continue
                
        
    # print(results)
    return results


def locate_event(event,segment,max_retries=4, base_delay=4):
    system_prompt='''
Your are an expert of finding a specific event in a video. Given the event description and the video, you should find the time range of the event in the video. 
### Guidelines for Locating Event
- You should give the time range in seconds.
### Output format
Your output should look like this: 
"BEGIN_TIME: <begin_second>, END_TIME: <end_second>"
    '''

    prompt = f'Locate the time range of the event: {event}.'
    messages= [
            {
                "role": "system",
                "content": [{"type":'text', "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type":"video", "video": segment,"fps":fps},
                    {"type":"text", "text": prompt}
                ],
            }
        ]
    for attempt in range(max_retries):
        try:
            response = dashscope.MultiModalConversation.call(model='qwen2.5-vl-7b-instruct', messages=messages)
            if response==None:
                time.sleep(base_delay * (attempt + 1))
                continue
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(base_delay * (attempt + 1))
        
    print(event,response.output.choices[0].message.content[0]["text"])
    max_lenth=get_video_duration(segment)
    try:
        return parsing_time(event,response.output.choices[0].message.content[0]["text"],max_lenth=max_lenth)
    except Exception as e:
        return {'BEGIN_TIME':0,'END_TIME':100, 'event':event}

def parsing_time(event,time,max_lenth=30.):
    time_pattern = r'(BEGIN_TIME|END_TIME)\s*:\s*(\d+\.?\d*|\d*\.?\d+)'
    
    matches = re.findall(time_pattern, time)
    time_dict = {key: float(value) for key, value in matches}
    time_dict['event']=event
    if(time_dict['END_TIME']>max_lenth):
        reduction = time_dict['END_TIME'] - max_lenth
        time_dict['END_TIME'] = max_lenth 
        time_dict['BEGIN_TIME'] -= reduction 
        if time_dict['BEGIN_TIME']<0:
            time_dict['BEGIN_TIME'] = 0

    # print(time_dict)
    if not ('BEGIN_TIME' in time_dict and 'END_TIME' in time_dict):
        raise ValueError("Invalid time format: missing BEGIN_TIME or END_TIME")
    return time_dict
    

if __name__ == '__main__':
    video_id='0ARh-9Amrv0'
    
    visual_events(video_id)