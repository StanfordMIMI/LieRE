#!/bin/bash
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
POSITION_ENCODING="liere"
MODEL_SIZE="base"
NAME="ucf101"
RUN_NAME="${NAME}_${POSITION_ENCODING}_${MODEL_SIZE}_${TIMESTAMP}"

python lightning_main.py fit \
    --config lightning_conf/base.yaml \
    --config lightning_conf/ucf101.yaml \
    --trainer.devices 1 \
    --trainer.accumulate_grad_batches 2 \
    --trainer.check_val_every_n_epoch 5 \
    --trainer.logger.init_args.name $RUN_NAME \
    --trainer.callbacks.init_args.dirpath "/dataNAS/people/sostm/checkpoints/${RUN_NAME}" \
    --model.init_args.model_architecture $POSITION_ENCODING \
    --model.init_args.model_size $MODEL_SIZE \
    --data.init_args.per_device_batch_size 4 \
    --data.init_args.data_dir /dataNAS/people/sostm/data/sostm_ucf101 \
    --model.init_args.shuffle_patches False \
    --model.init_args.rotary_embedding_per_layer True \
    --model.init_args.rotary_embedding_per_head True \
    --model.init_args.checkpoint_attn False \

python lightning_main.py validate \
    --config lightning_conf/base.yaml \
    --config lightning_conf/ucf101.yaml \
    --trainer.devices 1 \
    --trainer.accumulate_grad_batches 1 \
    --trainer.logger.init_args.name "${RUN_NAME}_val" \
    --model.init_args.model_architecture $POSITION_ENCODING \
    --model.init_args.model_size $MODEL_SIZE \
    --data.init_args.per_device_batch_size 64 \
    --data.init_args.data_dir /dataNAS/people/sostm/data/sostm_ucf101 \
    --model.init_args.shuffle_patches False \
    --model.init_args.rotary_embedding_per_layer True \
    --model.init_args.rotary_embedding_per_head True \
    --model.init_args.checkpoint_attn False \
    --model.init_args.generator_dim 64 \
    --ckpt "/dataNAS/people/sostm/checkpoints/${RUN_NAME}/last.ckpt" \
