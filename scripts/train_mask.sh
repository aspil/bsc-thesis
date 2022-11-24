#!/bin/bash


if [ -z ${1+x} ]; then echo "era not set" && exit 1; else echo "data dir = $1_data_bin"; fi
if [ -z ${2+x} ]; then echo "architecture not set" && exit 1; else echo "architecture = $2"; fi

ERA=$1
NN_ARCH=$2
DATA_BIN_DIR=${ERA}_data_bin
NN_ARCH="musicbert_${NN_ARCH}"


# Hyperparameters
TOTAL_UPDATES=125000
WARMUP_UPDATES=25000
PEAK_LR=0.0005
TOKENS_PER_SAMPLE=8192
BATCH_SIZE=256
MAX_SENTENCES=4

N_GPU_LOCAL=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / ${N_GPU_LOCAL}))

# Checkpoint
CHECKPOINT_SUFFIX=${NN_ARCH}

echo $DATA_BIN_DIR
echo $NN_ARCH


fairseq-train ${DATA_BIN_DIR} --user-dir musicbert \
    --restore-file checkpoints/checkpoint_last_${CHECKPOINT_SUFFIX}.pt \
    --task masked_lm --criterion masked_lm \
    --arch ${NN_ARCH} --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} \
    --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 100 \
    --checkpoint-suffix _${ERA}_${CHECKPOINT_SUFFIX} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-interval-updates 1