#!/bin/bash

export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x32"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=448
export VIDEO_MINRES=288
export PAD2STRIDE=1
export FORCE_NO_DOWNSAMPLE=1
export LOAD_VISION_EARLY=1

export PYTHONPATH=/path/to/Ola:$PYTHONPATH

EXP_NAME="ola_7b_stage3"
DATA='/path/to/data.json'

CHECKPOINT='/path/to/Ola_Video'

echo $MASTER_ADDR; echo $nnode; echo $nrank

torchrun  --nproc_per_node 8 --nnodes=$nnode --node_rank=$nrank --master_addr=$MASTER_ADDR --master_port=12324 \
    ola/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --run_name $EXP_NAME \
    --model_name_or_path $CHECKPOINT \
    --pretrain_speech_projector $CHECKPOINT/speech_projector.bin \
    --vision_tower $VISION_TOWER \
    --mm_projector_type ola_mlp \
    --speech_projector_type "linear" \
    --mm_vision_select_layer -1 \
    --mm_use_im_patch_token False \
    --tune_speech_adapter False \
    --version qwen_1_5 \
    --data_path $DATA  \
    --bf16 True \
    --output_dir ./checkpoints/$EXP_NAME \
    --sample_independently True \
    --fix_speech_encoder True \
    --freeze_mm_vision_tower True \
    --speech_encoder "./pretrained/large-v3.pt" \
    --music_encoder "./pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    --speech_encoder_type "dual" \
    --speech_encoder_hidden_size 2048 \
    --speech_encoder_ds_rate 10 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1  \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --min_lr_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --frames_upbound 48 \
    --lazy_preprocess True \
    --report_to none