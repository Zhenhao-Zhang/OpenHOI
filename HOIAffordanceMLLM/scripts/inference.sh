#!/bin/bash

MODEL_VERSION=llama-vicuna-7b
TAG=8k

CUDA_VISIBLE_DEVICES=0 python llava/serve/cli.py \
    --model-path /root/autodl-tmp/checkpoints/$MODEL_VERSION-$TAG-finetune \
    --pts-file $1
    