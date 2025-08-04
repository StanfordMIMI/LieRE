#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
POSITION_ENCODING="liere"
MODEL_SIZE="base"
NAME="cifar100"
HEAD=True
LAYER=True
RUN_NAME="${NAME}_${POSITION_ENCODING}_${MODEL_SIZE}_${TIMESTAMP}"
GENERATOR=64
CKPT_NAME="${RUN_NAME}_${GENERATOR}"
SAVE_CHECKPOINTS="/dataNAS/people/sostm/checkpoints"


python lightning_main.py fit \
    --config lightning_conf/base.yaml \
    --config lightning_conf/cifar100.yaml \
    --trainer.devices 2 \
    --trainer.accumulate_grad_batches 2 \
    --trainer.logger.init_args.name $RUN_NAME \
    --trainer.callbacks.init_args.dirpath "${SAVE_CHECKPOINTS}/${RUN_NAME}" \
    --model.init_args.model_architecture $POSITION_ENCODING \
    --model.init_args.model_size $MODEL_SIZE \
    --model.init_args.freeze_liere False \
    --data.init_args.per_device_batch_size 128 \
    --data.ablation_factor 1 \
    --model.init_args.rotary_embedding_per_layer $LAYER \
    --model.init_args.rotary_embedding_per_head $HEAD \
    --model.init_args.generator_dim $GENERATOR

python lightning_main.py validate \
    --config lightning_conf/base.yaml \
    --config lightning_conf/cifar100.yaml \
    --trainer.devices 1 \
    --trainer.accumulate_grad_batches 2 \
    --trainer.logger.init_args.name $RUN_NAME \
    --trainer.callbacks.init_args.dirpath "${SAVE_CHECKPOINTS}/${RUN_NAME}" \
    --model.init_args.model_architecture $POSITION_ENCODING \
    --model.init_args.model_size $MODEL_SIZE \
    --model.init_args.freeze_liere False \
    --data.init_args.per_device_batch_size 128 \
    --data.ablation_factor 1 \
    --model.init_args.rotary_embedding_per_layer $LAYER \
    --model.init_args.rotary_embedding_per_head $HEAD \
    --ckpt "${SAVE_CHECKPOINTS}/${CKPT_NAME}/last.ckpt"
