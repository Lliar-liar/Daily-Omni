from openai import OpenAI
import os
import re
import json
import google.generativeai as genai
import random
import time
from config import SEGMENT_DURATION,MAX_SEGMENTS,GEMINI_API_KEY_OPTIMIZATION,DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DEEPSEEK_QA_MODEL,QA_OPTIMIZATION_SYSTEM_PROMPT
genai.configure(api_key=GEMINI_API_KEY_OPTIMIZATION,transport='rest') 
output_path=f"QAs_revise_{SEGMENT_DURATION*MAX_SEGMENTS}s.json"
system_prompt=QA_OPTIMIZATION_SYSTEM_PROMPT


def optimize_QA(caption_path,file_lock=None,max_retries=5, base_delay=1):
    if not os.path.exists(caption_path): # 如果目录不存在，则跳过
        return False
    video_id=os.path.basename(caption_path)
    if not os.path.exists(os.path.join(caption_path,video_id+'_video.mp4')): 
        return False
    video_caption_path=os.path.join(caption_path,'video_consistent_captions.txt')
    dir_name = os.path.basename(caption_path)
    audio_caption_path=os.path.join(caption_path,"audio_revised_captions.txt")
    alignment_caption_path=os.path.join(caption_path,'av_alignment_captions.txt')
    try:
        with open(video_caption_path, 'r', encoding='utf-8') as f:
            video_content = f.read()
        with open(audio_caption_path, 'r', encoding='utf-8') as f:
            audio_content = f.read()
        with open(alignment_caption_path, 'r', encoding='utf-8') as f:
            alignment_content = f.read()
    except Exception as e:
        print("Error reading file:", e)
        return False
    if not os.path.exists(os.path.join(caption_path, 'QAs_advance_deepseek.json')):
        return False
    with open(os.path.join(caption_path, 'QAs_advance_deepseek.json'), 'r',encoding='utf-8') as f:
        original_QA = json.load(f)
    with open(os.path.join(caption_path, output_path), 'w', encoding='utf-8') as f:
        f.write('[]')
    original_QA_copy=original_QA.copy()
    for i, qa_pair in enumerate(original_QA):
        question = qa_pair.get("Question", "No question provided")
        choices = qa_pair.get("Choice", [])
        answer = qa_pair.get("Answer", "No answer provided")
        qa_type=qa_pair.get("Type", "No type provided")
        content_parent_category=qa_pair.get("content_parent_category", "No content_parent_category provided")
        content_fine_category=qa_pair.get("content_fine_category", "No content_fine_category provided")

        # Example processing or output for each question
        print(f"Processing QA Pair {i + 1}:")
        original_QA=f'''
Question: {question}
Choices: {choices}
Answer: {answer}
        '''
        content=f'''
Please optimize the question-answer pair:
## Original Question-Answer Pair:
{original_QA}
## Video Captions
The visual and audio description for the related video is shown as follows:
### Visual Caption:
{video_content}
### Audio Caption:
{audio_content}
### Audio Visual Event Happened at the same time(Audio_Event -- Visual_Event):
{alignment_content}
        '''
        print(original_QA)
        # print(content)

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=DEEPSEEK_QA_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    stream=False,
                    temperature=1,
    
                )
                break
            except Exception as e:
                if '429' in str(e):
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Received 429 error. Retrying in {delay:.2f} seconds (Attempt {attempt + 1}/{max_retries}).")
                    time.sleep(delay)
                else:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"An error occurred during API call (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f} seconds (Attempt {attempt + 1}/{max_retries}).")
                    time.sleep(delay)
        
        # print(response.choices[0].message.content)
        try:
            qa_pairs = json.loads(response.choices[0].message.content.removeprefix('```json').removesuffix('```'))
        except json.JSONDecodeError as e:
            print("JSON parse error: ", e)
            return False
        for qa_pair in qa_pairs:
            qa_pair["video_id"] = dir_name
            qa_pair['Type']=qa_type
            qa_pair["content_parent_category"]=content_parent_category
            qa_pair["content_fine_category"]=content_fine_category
        print(qa_pairs)
        json_path=os.path.join(caption_path, output_path)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8',errors='replace') as f:
                existing_data = json.load(f)
                existing_data.extend(qa_pairs)
                qa_pairs = existing_data
            with open(json_path, 'w', encoding='utf-8',errors='replace') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        else:
            with open(json_path, 'w', encoding='utf-8',errors='replace') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        
        
        
    return True
        
if __name__ == '__main__':
    optimize_QA()