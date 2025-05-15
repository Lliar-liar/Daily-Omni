import dashscope
import os
import google.generativeai as genai
import base64
import time
import random
import pathlib
import requests
from openai import OpenAI
from variables import base_path, dashscope_apikey,openai_apikey,openai_baseurl

dashscope.api_key = dashscope_apikey

def get_audio_path(video_id, base_path=base_path):
    audio_path=[]
    for i in range(0,3):
        if os.path.exists(os.path.join(base_path, video_id, f'{video_id}_audio_{i}.wav')):
            audio_path.append(os.path.join(base_path, video_id, f'{video_id}_audio_{i}.wav'))
        else: 
            break
    # print(video_path)
    return audio_path

def encode_audio(audio_file_path):
    """Encodes an audio file to base64."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Audio file not found: {audio_file_path}")
        return None

def get_audio_caption(video_id='0loP4WNnL2k',max_retries=4,base_delay=2,use_model="qwen2_audio"):
    audio_path = get_audio_path(video_id)
    results=''
    system_prompt='''
Describe the audio clip's sounds chronologically without mentioning specific timestamps. 
For speech, mention:
- who is speaking
- speaker's tone
For music, mention:
- music genre
- music tone
- instruments used
For other sounds, mention:
- Sound type (describe the sound)
## Output Format:
First, <Sound_1_Description>. Then, <Sound_2_Description>. Finally, <Sound_3_Description>.
'''

    if use_model=="qwen2_audio":
        for segment in audio_path:
            audio_base64=encode_audio(segment)
            

            messages = [
                {
                    "role": "system",
                    "content":"You are a helpful assistant.",
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
                "temperature": 0.7
            }
            for attempt in range(max_retries):
                try:
                    response = requests.get("http://127.0.0.1:6006/v1/audio/inference",json=payload)
                    print(response.json())
                    break
                except Exception as e:
                    print(f"Error while getting audio caption: {e}")
                    time.sleep(base_delay * (2 ** attempt))
                    continue
            # print(response)
            text_content = response.json()['response']
            results+=text_content+"\n"
    else:   
        for segment in audio_path:
            messages = [
                {
                    "role": "system",
                    "content": [{"text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"audio": segment},
                        {"text": "Describe the sounds in the audio clip."}
                    ],
                }
            ]
            for attempt in range(max_retries):
                try:
                    response = dashscope.MultiModalConversation.call(model="qwen-audio-turbo-1204", messages=messages)
                    break
                except Exception as e:
                    print(f"Error while getting audio caption: {e}")
                    time.sleep(base_delay * (2 ** attempt))
                    continue
            # print(response)
            text_content = response.output.choices[0].message.content[0]["text"]
            results+=text_content+"\n"
            print(text_content)
    print(results)
    return results

def get_speech(video_id='0x82_HySIVU',max_retries=4,base_delay=2):
    audio_path = get_audio_path(video_id)
    results=''
    for segment in audio_path:
        client = OpenAI(api_key=openai_apikey, base_url=openai_baseurl)
        audio_file= open(segment, "rb")
        for attempt in range(max_retries):
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
                results+=(transcription.text+"\n")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(base_delay * (2 ** attempt))
                continue
    print(results)
    return results
if __name__ == "__main__":
    # get_speech('0cZQm65sZjc')
    get_audio_caption()