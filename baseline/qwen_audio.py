
from fastapi import FastAPI, Request, File, UploadFile, Query, Body
import uvicorn
import json
import datetime
import torch
import os
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, GenerationConfig, AutoModel
from typing import List, Dict, Any
app = FastAPI()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

# model.generation_config.pad_token_id = model.generation_config.eos_token_id

@app.get("/v1/audio/inference")
async def audio_transcription(
    conversation: List[Dict[str, Any]] = Body(..., alias='conversation'), 
    temperature: float = Body(0.9, alias='temperature')
):
    global processor,model

    # print(conversation)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            BytesIO(urlopen(ele['audio_url']).read()),
                            sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True,sampling_rate=16000).to("cuda")
    # inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=1000)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return {"response": response}

if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1) 