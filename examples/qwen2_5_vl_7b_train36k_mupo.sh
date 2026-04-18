#!/bin/bash

set -x

# Variables
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
CONFIG="examples/config.yaml"
TRAIN_FILES="xytian1008/MUPO-Thinker-train36k@train"
VAL_FILES="xytian1008/MUPO-Thinker-val1k@test"
MAX_PROMPT_LENGTH=4096
ROLLOUT_BATCH_SIZE=384
GLOBAL_BATCH_SIZE=128
FILTER_OVERLONG_PROMPTS_WORKERS=128
TORCH_DTYPE=bf16
CLIP_RATIO_LOW=0.20
CLIP_RATIO_HIGH=0.28
MICRO_BATCH_SIZE_UPDATE=4
MICRO_BATCH_SIZE_EXPERIENCE=16
OPTIM_LR=1.0e-6
OPTIM_STRATEGY=adamw_bf16
TENSOR_PARALLEL_SIZE=1
ROLLOUT_N=5
REWARD_TYPE="batch"
REWARD_FUNCTION="examples/reward_function/mupo.py:compute_score"
EXPERIMENT_NAME="qwen2_5_vl_7b_mupo"
NGPUS_PER_NODE=8
VAL_BEFORE_TRAIN=true
VAL_ONLY=false
TOTAL_EPOCHS=5
VAL_FREQ=3
SAVE_FREQ=20
SAVE_LIMIT=-1
DISABLE_KL=true
ONLINE_FILTERING=true
ADV_ESTIMATOR="grpo"
PARTITION_K=1
PARTITION_MIN=5
BALANCE_BETA=1.0
LAMBDA_INIT=0.4
DECAY_RATE=9e-3

python3 -m verl.trainer.main \
    config=${CONFIG} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.fsdp.torch_dtype=${TORCH_DTYPE} \
    worker.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_EXPERIENCE} \
    worker.actor.optim.lr=${OPTIM_LR} \
    worker.actor.optim.strategy=${OPTIM_STRATEGY} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.reward.reward_type=${REWARD_TYPE} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.val_only=${VAL_ONLY} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_limit=${SAVE_LIMIT} \
    algorithm.disable_kl=${DISABLE_KL} \
    algorithm.online_filtering=${ONLINE_FILTERING} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.partition_k=${PARTITION_K} \
    algorithm.partition_min=${PARTITION_MIN} \
    algorithm.balance_beta=${BALANCE_BETA} \
    algorithm.lambda_init=${LAMBDA_INIT} \
    algorithm.decay_rate=${DECAY_RATE}