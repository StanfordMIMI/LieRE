#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
POSITION_ENCODING="liere"
MODEL_SIZE="base"
NAME="imagenet"
RUN_NAME="${NAME}_${POSITION_ENCODING}_${MODEL_SIZE}_${TIMESTAMP}"
CKPT_NAME="imagenet_liere_base_2024-11-24_23-23-31"
GENERATOR_DIMS=8

python lightning_main.py fit \
    --config lightning_conf/base.yaml \
    --config lightning_conf/imagenet_224.yaml \
    --data.init_args.per_device_batch_size 64 \
    --trainer.devices 8 \
    --trainer.accumulate_grad_batches 1 \
    --trainer.logger.init_args.name $RUN_NAME \
    --trainer.callbacks.init_args.dirpath "/dataNAS/people/sostm/checkpoints/${RUN_NAME}" \
    --model.init_args.model_architecture $POSITION_ENCODING \
    --model.init_args.model_size $MODEL_SIZE \
    --model.init_args.rotary_embedding_per_layer True \
    --model.init_args.rotary_embedding_per_head True \
    --model.init_args.imsize 224 \
    --data.init_args.imsize 224 \
    --ckpt "/dataNAS/people/sostm/checkpoints/${CKPT_NAME}/last.ckpt" \
    --model.init_args.generator_dim $GENERATOR_DIMS
