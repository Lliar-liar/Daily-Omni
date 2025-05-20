import json
import os
import cv2
import time
import random
import base64
import tqdm
import multiprocessing
from openai import OpenAI
import dashscope
from variables import base_path,dashscope_apikey,fps
dashscope.api_key = dashscope_apikey

def extract_frames(video_path, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Fail to open: {video_path}")
    
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

def video_to_base64_chunked(video_path, chunk_size=4096 * 1024):

    base64_string = ""  
    try:
        with open(video_path, "rb") as video_file:
            while True:
                chunk = video_file.read(chunk_size)
                if not chunk:
                    break 
                encoded_chunk = base64.b64encode(chunk)
                base64_string += encoded_chunk.decode('utf-8') 
    except FileNotFoundError:
        print(f"Error: Video file not found at {video_path}")
        return None
    except Exception as e:
        print(f"Error encoding video: {e}")
        return None

    return base64_string
def get_video_path(video_id, base_path=base_path):
    video_path=[]
    for i in range(0,3):
        if os.path.exists(os.path.join(base_path, video_id, f'{video_id}_video_{i}.mp4')):
            video_path.append(os.path.join(base_path, video_id, f'{video_id}_video_{i}.mp4'))
        else: 
            break
    # print(video_path)
    return video_path


def get_visual_caption(video_id, max_retries=5, base_delay=4):
    video_path = get_video_path(video_id)
    result=''
    for segment in video_path:

        system_prompt = f"""
### Task:
You are an expert in understanding scene transitions based on visual features in a video. You are requested to create the descriptions for the current clip sent to you,  which includes multiple sequential frames.
#### Guidelines For Clip Description:
- Analyze the narrative progression implied by the sequence of frames, interpreting the sequence as a whole. 
- If text appears in the frames, you must describe the text in its original language and provide an English translation in parentheses. Additionally, explain the meaning of the text within its context.
- When referring to people, use their characteristics, such as clothing, to distinguish different people.
- **IMPORTANT** Please provide as many details as possible in your description, including number, colors, shapes, and textures of objects, actions and characteristics of humans, as well as scenes and backgrounds. 
- Pay special attention to describing the differences between frames within the clip. Detail how objects, people, actions, scenes, and other visual elements change from one frame to the next.
#### Output format
Your description of the video should look like this:"In this section, ... Then, ... After that, ... Finally, ...." Your description should be in 1 line.
"""
        
 
        prompt = 'Describe the video in detail.'
        messages= [
            {
                "role": "system",
                "content": [{"type":'text', "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type":"video", "video": segment,"fps":fps},
                    {"type":"text", "text": "Describe the video in detail."}
                ],
            }
        ]
        # base64_images = extract_frames(segment, fps)
        # print(messages)
        for attempt in range(max_retries):
            try:
                response = dashscope.MultiModalConversation.call(model='qwen2.5-vl-7b-instruct', messages=messages)
                # print(response)
                result+=response.output.choices[0].message.content[0]["text"].replace("\n","")+"\n"
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(base_delay * (attempt + 1))
      
        
    print(result)
    return result

if __name__ == '__main__':
    video_id='Me4W36_lUcI'
    get_visual_caption(video_id)