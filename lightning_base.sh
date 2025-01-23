#!/bin/bash

if [ -z "$DATASET" ]; then
#    export DATASET="CIFAR100"
    exit -1
fi
echo "Training on ${DATASET}"

CONFIG_FILE="${HOME}/sky_workdir/lightning_conf/${DATASET}.yaml"
if [ ! -f $CONFIG_FILE ]; then
    echo "Error: Configuration file $CONFIG_FILE does not exist."
    exit 1
fi

CHECKPOINT_DIR="/checkpoints/${SKYPILOT_TASK_ID}/"
CHECKPOINT_FILE=$(find "$CHECKPOINT_DIR" -name "last.ckpt" -print -quit)


if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CHECKPOINT_FILE" ]; then
    echo "Restarting from checkpoint in ${CHECKPOINT_DIR}"
    python lightning_main.py fit --config lightning_conf/base.yaml --config lightning_conf/${DATASET}.yaml --trainer.default_root_dir "$CHECKPOINT_DIR" --trainer.resume_from_checkpoint "$CHECKPOINT_FILE"
else
    echo "Creating checkpoint directory ${CHECKPOINT_DIR}"
    mkdir -p "$CHECKPOINT_DIR"
    python lightning_main.py fit --config lightning_conf/base.yaml --config lightning_conf/${DATASET}.yaml --trainer.default_root_dir "$CHECKPOINT_DIR"
fi