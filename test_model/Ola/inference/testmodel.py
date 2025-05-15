import os
import sys
sys.path.append('./')
import argparse
import json
import tqdm
import os




os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

import gradio as gr
import torch
import re
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import transformers
import moviepy.editor as mp
from typing import Dict, Optional, Sequence, List
import librosa
import whisper
from ola.conversation import conv_templates, SeparatorStyle
from ola.model.builder import load_pretrained_model
from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token, tokenizer_speech_token
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
import argparse
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

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

def get_video_path(video_id, base_path):
    video_path = os.path.join(base_path, video_id, f'{video_id}_video.mp4')
    return video_path



def evaluate_answer(model_answer, correct_answer):
    """Compares the model's answer with the correct answer."""
    # Handle potential None or empty string from the model
    if not model_answer:
        return False
    return model_answer.upper().startswith(correct_answer.upper())



def test_all_questions(file_path, model, args):  # Add runner as an argument
    """Tests all questions in the file."""
    qa_type_count={}
    qa_type_correct={}
    video_cat_count={}
    video_cat_correct={}
    data = load_json_data(file_path)
    if not data:
        return
    total_questions = len(data)
    correct_answers = 0
    failed=0
    VIDEO_CAT=[]
    QA_TYPE=[]
    for i, item in enumerate(data):
        video_category=item.get('video_category')
        qa_type=item.get('Type')
        if video_category not in VIDEO_CAT:
            VIDEO_CAT.append(video_category)
        if qa_type not in QA_TYPE:
            QA_TYPE.append(qa_type)
    VIDEO_CAT.sort()
    QA_TYPE.sort()
    for qa_type in QA_TYPE:
        qa_type_count[qa_type]=0
        qa_type_correct[qa_type]=0
    for video_category in VIDEO_CAT:
        video_cat_count[video_category]=0
        video_cat_correct[video_category]=0
    # data = data[800:810]
    total_questions = len(data)
    correct_answers = 0
    failed = 0
    qa_duration_count={"30s":0, "60s":0}
    qa_duration_correct={"30s":0, "60s":0}
    

    

    for item in tqdm.tqdm(data, desc="Evaluating Questions"): # Add a description
        question = item.get('Question')
        choices = item.get('Choice')
        correct_answer = item.get('Answer')
        video_id = item.get('video_id')
        qa_type=item.get('Type')
        video_category=item.get('video_category')
        video_duration=item.get('video_duration')
        

        if not all([question, choices, correct_answer, video_id]):
            print(f"Warning: Skipping item, missing required fields. Item: {item}")
            continue

        video_path = get_video_path(video_id,args.video_base_path)
        prompt = f"""
Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.
Question: {question}
Choices: {choices}
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
"""

        try:  # Add a try-except block
            model_answer=ask_model(args,model,video_path,prompt)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            failed +=1 # increase failed counter.
            continue

        is_correct = evaluate_answer(model_answer, correct_answer)

        print(f"\nQuestion: {question}")
        print(f"  Model's Answer: {model_answer}")
        print(f"  Correct Answer: {correct_answer}")
        print(f"  Is Correct: {is_correct}")

     
            
        qa_type_count[qa_type]+=1
        video_cat_count[video_category]+=1
        qa_duration_count[video_duration]+=1
        
        if is_correct:
            correct_answers += 1
            qa_type_correct[qa_type]+=1
            video_cat_correct[video_category]+=1
            qa_duration_correct[video_duration]+=1
    print(f"\nAccuracy: {correct_answers}/{total_questions} = {correct_answers / total_questions:.2%}\n")
    for qa_type in QA_TYPE:
        if qa_type_count[qa_type]==0:
            print(f"{qa_type}: 0/0 = ---")
        else:
            print(f"{qa_type}: {qa_type_correct[qa_type]}/{qa_type_count[qa_type]} = {qa_type_correct[qa_type] / qa_type_count[qa_type]:.2%}")
    print('\n')
    for video_category in VIDEO_CAT:
        if video_cat_count[video_category]==0:
            print(f"{video_category}: 0/0 = ---")
        else:
            print(f"{video_category}: {video_cat_correct[video_category]}/{video_cat_count[video_category]} = {video_cat_correct[video_category] / video_cat_count[video_category]:.2%}")
    print("\n")
    if qa_duration_count['30s']!=0:
        print(f"30s  Duration: {qa_duration_correct['30s']}/{qa_duration_count['30s']} = {qa_duration_correct['30s'] / qa_duration_count['30s']:.2%}")
    if qa_duration_count['60s']!=0:
        print(f"60s  Duration: {qa_duration_correct['60s']}/{qa_duration_count['60s']} = {qa_duration_correct['60s'] / qa_duration_count['60s']:.2%}")
    print(f"Failed: {failed}")

def load_audio(audio_file_name):
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    speechs = []
    speech_wavs = []

    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(chunk)

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs

def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip.audio

def ask_model(args,model,video_path,text):

    image_path = args.image_path
    audio_path = args.audio_path

    if video_path is not None:
        modality = "video"
        visual = video_path
        assert image_path is None

    elif image_path is not None:
        visual = image_path
        modality = "image"
        assert video_path is None

    elif audio_path is not None:
        modality = "text"


    # input audio and video, do not parse audio in the video, else parse audio in the video
    # if audio_path:
    #     USE_SPEECH = True
    # elif modality == "video":
    #     USE_SPEECH = True
    # else:
    #     USE_SPEECH = False
    # USE_SPEECH=args.USE_SPEECH
    USE_SPEECH=True

    speechs = []
    speech_lengths = []
    speech_wavs = []
    speech_chunks = []
    if modality == "video":
        vr = VideoReader(visual, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 64, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video = [Image.fromarray(frame) for frame in spare_frames]
    elif modality == "image":
        image = [Image.open(visual)]
        image_sizes = [image[0].size]
    else:
        images = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        images_highres = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        image_sizes = [(224, 224)]


    if USE_SPEECH and audio_path:
        audio_path = audio_path
        speech, speech_length, speech_chunk, speech_wav = load_audio(audio_path)
        speechs.append(speech.bfloat16().to('cuda'))
        speech_lengths.append(speech_length.to('cuda'))
        speech_chunks.append(speech_chunk.to('cuda'))
        speech_wavs.append(speech_wav.to('cuda'))
        print('load audio')
    elif USE_SPEECH and not audio_path:
        # parse audio in the video
        audio = extract_audio(visual)
        audio.write_audiofile("./video_audio.wav")
        video_audio_path = './video_audio.wav'
        speech, speech_length, speech_chunk, speech_wav = load_audio(video_audio_path)
        speechs.append(speech.bfloat16().to('cuda'))
        speech_lengths.append(speech_length.to('cuda'))
        speech_chunks.append(speech_chunk.to('cuda'))
        speech_wavs.append(speech_wav.to('cuda'))
    else:
        speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
        speech_lengths = [torch.LongTensor([3000]).to('cuda')]
        speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
        speech_chunks = [torch.LongTensor([1]).to('cuda')]

    conv_mode = "qwen_1_5"
    if text:
        qs = text
    else:
        qs = ''

    if USE_SPEECH and audio_path and image_path: # image + speech instruction
        qs = DEFAULT_IMAGE_TOKEN + "\n" + "User's question in speech: " + DEFAULT_SPEECH_TOKEN + '\n'
    elif USE_SPEECH and video_path: # video + audio
        qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif USE_SPEECH and audio_path: # audio + text
        qs = DEFAULT_SPEECH_TOKEN + "\n" + qs
    elif image_path or video_path: # image / video
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif text: # text
        qs = qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if USE_SPEECH and audio_path and image_path: # image + speech instruction
        input_ids = tokenizer_speech_question_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    elif USE_SPEECH and video_path: # video + audio
        input_ids = tokenizer_speech_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    elif USE_SPEECH and audio_path: # audio + text
        input_ids = tokenizer_speech_token(prompt, tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    else:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')

    if modality == "video":
        video_processed = []
        for idx, frame in enumerate(video):
            image_processor.do_resize = False
            image_processor.do_center_crop = False
            frame = process_anyres_video(frame, image_processor)

            if frame_idx is not None and idx in frame_idx:
                video_processed.append(frame.unsqueeze(0))
            elif frame_idx is None:
                video_processed.append(frame.unsqueeze(0))
        
        if frame_idx is None:
            frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
        
        video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
        video_processed = (video_processed, video_processed)

        video_data = (video_processed, (384, 384), "video")
    elif modality == "image":
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        image_tensor, image_highres_tensor = [], []
        for visual in image:
            image_tensor_, image_highres_tensor_ = process_anyres_highres_image(visual, image_processor)
            image_tensor.append(image_tensor_)
            image_highres_tensor.append(image_highres_tensor_)
        if all(x.shape == image_tensor[0].shape for x in image_tensor):
            image_tensor = torch.stack(image_tensor, dim=0)
        if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
            image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
        if type(image_tensor) is list:
            image_tensor = [_image.bfloat16().to("cuda") for _image in image_tensor]
        else:
            image_tensor = image_tensor.bfloat16().to("cuda")
        if type(image_highres_tensor) is list:
            image_highres_tensor = [_image.bfloat16().to("cuda") for _image in image_highres_tensor]
        else:
            image_highres_tensor = image_highres_tensor.bfloat16().to("cuda")

    pad_token_ids = 151643

    attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    gen_kwargs = {}

    if "max_new_tokens" not in gen_kwargs:
        gen_kwargs["max_new_tokens"] = 1024
    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0.2
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1

    with torch.inference_mode():
        if modality == "video":
            output_ids = model.generate(
                inputs=input_ids,
                images=video_data[0][0],
                images_highres=video_data[0][1],
                modalities=video_data[2],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )
        elif modality == "image":
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor,
                images_highres=image_highres_tensor,
                image_sizes=image_sizes,
                modalities=['image'],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )
        elif modality == "text":
            output_ids = model.generate(
                input_ids,
                images=images,
                images_highres=images_highres,
                image_sizes=image_sizes,
                modalities=['text'],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='THUdyh/Ola-7b')
    parser.add_argument('--text', type=str, default='Describe the video')
    parser.add_argument('--audio_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument(
        '--video_base_dir',
        type=str,
        default='Videos', # Added default from original global var
        help='Base directory containing video folders.'
    )
    parser.add_argument(
        '--json_file_path',
        type=str,
        default='qa.json', # Added default from original global var
        help='Path to the JSON file containing QA pairs.'
    )
    # parser.add_argument('--USE_SPEECH', type=bool, default=False,help="False for no audio, True for with audio")
    # parser.add_argument('--use_audio', action='store_true', help="Include audio in processing")
    args = parser.parse_args()

    model_path = args.model_path
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None)
    model = model.to('cuda').eval()
    model = model.bfloat16()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_all_questions(args.json_file_path, model, args)

    # ask_model(args,model)